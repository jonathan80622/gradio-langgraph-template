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

logger = logging.getLogger(__name__)
ASSISTANT_SYSTEM_PROMPT = """You are a helpful assistant.

Use download_website_text to download the text from a website.
"""

@tool
async def download_website_text(url: str) -> str:
    """Downloads the text from a website"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                downloaded = await response.text()
        result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', with_metadata=True)
        return result or "No text found on the website"
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return f"Error retrieving website content: {str(e)}"

tools = [download_website_text]

if os.environ.get("TAVILY_API_KEY"):
    tools.append(
        TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
        )
    )
    ASSISTANT_SYSTEM_PROMPT += "Use tavily_search_results_json if to search for relevant information online."
else:
    print("TAVILY_API_KEY environment variable not found. Websearch disabled")

model = ChatOpenAI(model="gpt-4o-mini", tags=["assistant"])
assistant_model = model.bind_tools(tools)

class GraphProcessingState(BaseModel):
    # user_input: str = Field(default_factory=str, description="The original user input")
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

async def assistant_node(state: GraphProcessingState, config=None):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASSISTANT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | assistant_model
    response = await chain.ainvoke({"messages": state.messages}, config=config)

    return {
        "messages": response
    }

def assistant_cond_edge(state: GraphProcessingState):
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"Tool call detected: {last_message.tool_calls}")
        return "tools"
    return END

def define_workflow() -> CompiledStateGraph:
    """Defines the workflow graph"""
    # Initialize the graph
    workflow = StateGraph(GraphProcessingState)

    # Add nodes
    workflow.add_node("assistant_node", assistant_node)
    workflow.add_node("tools", ToolNode(tools))

    # Edges
    workflow.add_edge("tools", "assistant_node")

    # Conditional routing
    workflow.add_conditional_edges(
        "assistant_node",
        # If the latest message (result) from assistant is a tool call -> assistant_cond_edge routes to tools
        # If the latest message (result) from assistant is a not a tool call -> assistant_cond_edge routes to END
        assistant_cond_edge,
    )
    # Set end nodes
    workflow.set_entry_point("assistant_node")
    # workflow.set_finish_point("assistant_node")

    return workflow.compile()

graph = define_workflow()
