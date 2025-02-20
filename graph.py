from typing import Annotated

import aiohttp
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig
from langgraph.graph import Graph, StateGraph, END, MessagesState, add_messages
from pydantic import BaseModel, Field, ValidationError
from trafilatura import extract

@tool
async def download_website_text(url: str, config: RunnableConfig) -> str:
    """Downloads the text from a website

    args:
        url: The URL of the website
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            downloaded = await response.text()
    result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', with_metadata=True)
    if result:
        return result
    return "No text found on the website"

tools = [download_website_text]

ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."
assistant_model = ChatOpenAI(model="gpt-4o-mini", tags=["assistant"]).bind_tools(tools)

class GraphProcessingState(BaseModel):
    user_input: str = Field(default_factory=str, description="The original user input")
    history: list[dict] = Field(default_factory=list, description="Chat history")  # type: ignore
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


async def assistant_node(state: GraphProcessingState, config=None):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ASSISTANT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{user_input}"),
            *state.messages,
        ]
    )
    chain = prompt | assistant_model
    response = await chain.ainvoke({"user_input": state.user_input, "chat_history": state.history}, config)
    return {"messages": response}

def assistant_cond_edge(state: GraphProcessingState, config=None):
    if not state.messages[-1].content:
        return "tools"
    return END

def define_workflow() -> CompiledStateGraph:
    """Defines the workflow graph"""
    # Initialize the graph
    workflow = StateGraph(GraphProcessingState)

    # Add nodes
    workflow.add_node("assistant_node", assistant_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge("tools", "assistant_node")

    # Conditional routing
    workflow.add_conditional_edges(
        "assistant_node",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        assistant_cond_edge,
    )
    # Set end nodes
    workflow.set_entry_point("assistant_node")
    # workflow.set_finish_point("assistant_node")

    return workflow.compile()

graph = define_workflow()
#
# async def process_user_input_graph(input_state: GraphProcessingState, thread_id=None) -> GraphProcessingState:
#     config: RunnableConfig = RunnableConfig()
#     if "configurable" not in config:
#         config["configurable"] = {}
#     if thread_id:
#         config["configurable"]["thread_id"] = thread_id
#     final_state_dict = await graph.ainvoke(
#         input_state,
#         config=config,
#     )
#     final_state = GraphProcessingState(**final_state_dict)
#     final_state.user_input = ""
#     final_state.history = []
#     return final_state
