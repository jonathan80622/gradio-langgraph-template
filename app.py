#!/usr/bin/env python

import logging
import logging.config
from typing import Any
from uuid import uuid4, UUID
import json

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig
from pydantic import BaseModel

load_dotenv()

# There are tools set here dependent on environment variables
from graph import graph, model # noqa

FOLLOWUP_QUESTION_NUMBER = 3
TRIM_MESSAGE_LENGTH = 16  # Includes tool messages
USER_INPUT_MAX_LENGTH = 1000  # Characters

with open('logging-config.json', 'r') as fh:
    config = json.load(fh)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

async def chat_fn(user_input: str, history: dict, input_graph_state: dict, uuid: UUID):
    """
    Args:
        user_input (str): The user's input message
        history (dict): The history of the conversation in gradio
        input_graph_state (dict): The current state of the graph. This includes tool call history
        uuid (UUID): The unique identifier for the current conversation. This can be used in conjunction with langgraph or for memory
    Yields:
        str|Any: The output message
        dict|Any: The final state of the graph
        bool|Any: Whether to trigger follow up questions

        We do not use gradio history in the graph since we want the ToolMessage in the history
        ordered properly. GraphProcessingState.messages is used as history instead
    """
    try:
        if "messages" not in input_graph_state:
            input_graph_state["messages"] = []
        input_graph_state["messages"].append(
            HumanMessage(user_input[:USER_INPUT_MAX_LENGTH])
        )
        input_graph_state["messages"] = input_graph_state["messages"][-TRIM_MESSAGE_LENGTH:]
        config = RunnableConfig()
        config["configurable"] = {}
        config["configurable"]["thread_id"] = uuid

        output: str = ""
        final_state: dict | Any = {}
        waiting_output_seq: list[str] = []

        yield "Processing...", gr.skip(), False

        async for stream_mode, chunk in graph.astream(
                    input_graph_state,
                    config=config,
                    stream_mode=["values", "messages"],
                ):
            if stream_mode == "values":
                final_state = chunk
            elif stream_mode == "messages":
                msg, metadata = chunk
                # download_website_text is the name of the function defined in graph.py
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for msg_tool_call in msg.tool_calls:
                        tool_name: str = msg_tool_call['name']
                        if tool_name == "download_website_text":
                            waiting_output_seq.append("Downloading website text...")
                            yield "\n".join(waiting_output_seq), gr.skip(), gr.skip()
                        elif tool_name == "tavily_search_results_json":
                            waiting_output_seq.append("Searching for relevant information...")
                            yield "\n".join(waiting_output_seq), gr.skip(), gr.skip()
                        elif tool_name:
                            waiting_output_seq.append(f"Running {tool_name}...")
                            yield "\n".join(waiting_output_seq), gr.skip(), gr.skip()

                # print("output: ", msg, metadata)
                # assistant_node is the name we defined in the langgraph graph
                if metadata['langgraph_node'] == "assistant_node" and msg.content:
                    output += msg.content
                    yield output, gr.skip(), gr.skip()
        # Trigger for asking follow up questions
        # + store the graph state for next iteration
        yield output, dict(final_state), True
    except Exception:
        logger.exception("Exception occurred")
        user_error_message = "There was an error processing your request. Please try again."
        yield user_error_message, gr.skip(), False

def clear():
    return dict(), uuid4()

class FollowupQuestions(BaseModel):
    """Model for langchain to use for structured output for followup questions"""
    questions: list[str]

async def populate_followup_questions(end_of_chat_response, messages):
    """
    This function gets called a lot due to the asyncronous nature of streaming

    Only populate followup questions if streaming has completed and the message is coming from the assistant
    """
    if not end_of_chat_response or not messages:
        return [gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)]
    if messages[-1]["role"] == "assistant":
        follow_up_questions: FollowupQuestions = await model.with_structured_output(FollowupQuestions).ainvoke([
            ("system", f"suggest {FOLLOWUP_QUESTION_NUMBER} followup questions for the user to ask the assistant. Refrain from asking personal questions."),
            *messages,
        ])
        if len(follow_up_questions.questions) != FOLLOWUP_QUESTION_NUMBER:
            raise ValueError("Invalid value of followup questions")
        buttons = []
        for i in range(FOLLOWUP_QUESTION_NUMBER):
            buttons.append(
                gr.Button(follow_up_questions.questions[i], visible=True, elem_classes="chat-tab"),
            )
        return buttons
    else:
        return [gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)]

CSS = """
footer {visibility: hidden}
.followup-question-button {font-size: 12px }
"""

if __name__ == "__main__":
    logger.info("Starting the interface")
    with gr.Blocks(title="Langgraph Template", fill_height=True, css=CSS) as app:
        uuid_state = gr.State(
            uuid4
        )
        gradio_graph_state = gr.State(
            lambda: dict()
        )
        end_of_chat_response_state = gr.State(
            lambda: bool()
        )
        chatbot = gr.Chatbot(
            type="messages",
            scale=1,
        )
        chatbot.clear(fn=clear, outputs=[gradio_graph_state, uuid_state])
        with gr.Row():
            followup_question_buttons = []
            for i in range(FOLLOWUP_QUESTION_NUMBER):
                btn = gr.Button(f"Button {i+1}", visible=False, elem_classes="followup-question-button")
                followup_question_buttons.append(btn)

        chat_interface = gr.ChatInterface(
            chatbot=chatbot,
            fn=chat_fn,
            additional_inputs=[
                gradio_graph_state,
                uuid_state,
            ],
            additional_outputs=[
                gradio_graph_state,
                end_of_chat_response_state
            ],
            type="messages",
            multimodal=False,
        )

        def click_followup_button(btn):
            buttons = [gr.Button(visible=False) for _ in range(len(followup_question_buttons))]
            return btn, *buttons
        for btn in followup_question_buttons:
            btn.click(fn=click_followup_button, inputs=[btn], outputs=[chat_interface.textbox, *followup_question_buttons])

        chatbot.change(fn=populate_followup_questions, inputs=[end_of_chat_response_state, chatbot], outputs=followup_question_buttons, trigger_mode="once")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
