import logging
import os
from typing import Annotated

import aiohttp
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel, Field
from trafilatura import extract

import asyncio
import pickle, json
from typing_extensions import TypedDict
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_aws.chat_models.bedrock import ChatBedrock

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)
ASSISTANT_SYSTEM_PROMPT_BASE = """"""
search_enabled = bool(os.environ.get("TAVILY_API_KEY"))

class RagState(TypedDict):
    query: str
    retrieved: list[str]
    reranked: list[str]
    response: str
tools = [download_website_text]
rag_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

async def retrieve_node(state: RagState) -> dict:
    docs = await retriever.retrieve(state["query"], k=K)
    return {"retrieved": docs}

async def rerank_node(state: RagState) -> dict:
    prompt = (
        "You are a relevance ranking engine. Rank each document from 0.0 to 1.0 by relevance.\n\n"
        f"Query: {state['query']}\nDocuments:\n" +
        "\n".join(f"- {d}" for d in state["retrieved"])
    )
    response = await rag_llm.ainvoke(
        [HumanMessage(content=prompt)],
        tools=[{
            "name": "RankingOutput",
            "description": "Ranks docs by relevance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc": {"type": "string"},
                                "score": {"type": "number"}
                            },
                            "required": ["doc", "score"]
                        }
                    }
                },
                "required": ["items"]
            }
        }],
        tool_choice={"tool": {"name": "RankingOutput"}}
    )
    args = response.tool_calls[0]["args"]
    if isinstance(args, str):
        args = json.loads(args)
    top_docs = [item["doc"] for item in args["items"]]
    return {"reranked": top_docs}

async def respond_node(state: RagState) -> dict:
    context = "\n".join(state["reranked"])
    prompt = (
        f"Answer the following user query using the provided ranked context.\n\n"
        f"Query: {state['query']}\nContext:\n{context}"
    )
    answer_chunks = []
    async for chunk in rag_llm.astream([HumanMessage(content=prompt)]):
        answer_chunks.append(chunk.content)
    answer = "".join(answer_chunks)
    return {"response": answer}

def build_async_rag_graph():
    g = StateGraph(RagState)
    g.set_entry_point("retrieve")
    g.add_node("retrieve", RunnableLambda(retrieve_node))
    g.add_node("rerank",   RunnableLambda(rerank_node))
    g.add_node("respond",  RunnableLambda(respond_node))
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank",   "respond")
    g.add_edge("respond",  END)
    return g.compile()

@tool
async def rag_tool(query: str) -> str:
    """Construct a good query to etrieve internal database in order to find better information to answer question"""
    print('Rag tool invoked, ingesting', query)
    async_rag_graph = build_async_rag_graph()
    result = await async_rag_graph.ainvoke({"query": query})
    return result["response"]

# default docstring = A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.
tavily_search = TavilySearch(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
)


from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class MainState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # instead of replace API, use delta API

assistant_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

async def tool_selector_node(state: MainState) -> dict:
    system_msg = SystemMessage(
        content=(
            """
            You are an intelligent assistant with access to two tools:

            1. rag_tool(query: string) → “Retrieve relevant context from our internal database.”
            2. tavily_search(query: string) → “Search the web for the latest information.”

            RULES FOR TOOL USAGE:
            - If the user’s question requires external data (e.g., “What is the current state of EEMI?”, “Find the latest news on [topic]”), you MUST call exactly one function by outputting JSON—nothing else.
            - If you can answer the user’s question without external retrieval (e.g., simple factual or definitional questions), respond directly in plain text (no tool call).

            JSON FORMAT FOR A TOOL CALL (copy exactly):
        ```json
        {{ "name": "rag_tool", "arguments": {{ "query": "YOUR QUERY HERE" }} }}
        ```
        or
        ```json
        {{ "name": "tavily_search", "arguments": {{ "query": "YOUR QUERY HERE" }} }}
        ```
            """
        )
    )
    # (b) Pass [system_msg, *human_messages] into the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg.content),
        MessagesPlaceholder(variable_name="messages"), # iterable of messages
    ])
    chain = prompt | assistant_llm.bind_tools([rag_tool, tavily_search])

    response = await chain.ainvoke({"messages": state["messages"]}) # response is one AIMessage, not a singleton list of AIMessage

    return {"messages": [response]}

def tool_router(state: MainState) -> list[str]:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return [END]

    targets: list[str] = []
    for tool_call in last.tool_calls:
        print("In tool_router found:", tool_call)
        if tool_call["name"] == "rag_tool":
            targets.append("rag_node")
        elif tool_call["name"] == "tavily_search":
            targets.append("tavily_node")

    # If no known tool names found, just end
    return targets if targets else [END]


async def info_gatherer_node(state: MainState) -> dict:
    system_msg = SystemMessage(
        content="Generate a helpful response to the user based on the available messages and any gathered context."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg.content),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | assistant_llm

    response = await chain.ainvoke({"messages": state["messages"]})
    return {"messages": [response]}


def define_workflow() -> CompiledStateGraph:
    """Defines the workflow graph"""
    builder = StateGraph(MainState)
    builder.set_entry_point("tool_selector")
    
    builder.add_node("tool_selector", RunnableLambda(tool_selector_node))
    builder.add_node("rag_node", ToolNode([rag_tool]))
    builder.add_node("tavily_node", ToolNode([tavily_search]))
    builder.add_node("info_gatherer", RunnableLambda(info_gatherer_node))
    
    builder.add_conditional_edges(
        "tool_selector",
        tool_router,
        {"rag_node": "rag_node", "tavily_node": "tavily_node", END: END}
    )
    
    builder.add_edge("rag_node", "info_gatherer")
    builder.add_edge("tavily_node", "info_gatherer")
    builder.add_edge("info_gatherer", END)
    
    return builder.compile()

graph = define_workflow()
