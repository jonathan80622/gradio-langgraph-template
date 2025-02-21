#!/usr/bin/env python

from uuid import uuid4
import logging
import logging.config
import json

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import RunnableConfig

from graph import GraphProcessingState, graph

logger = logging.getLogger(__name__)

def setup_logging():
    with open("logging-config.json") as fh:
        config = json.load(fh)
    logging.config.dictConfig(config)

async def chat_fn(message, history, input_graph_state, uuid):
    try:
        input_graph_state.user_input = message
        input_graph_state.history = history
        config = RunnableConfig()
        config["configurable"] = {}
        config["configurable"]["thread_id"] = uuid

        output = ""
        async for msg, metadata in graph.astream(
                    {"user_input": input_graph_state.user_input, "history": input_graph_state.history},
                    config=config,
                    stream_mode="messages",
                ):
            # download_website_text is the name of the function defined in graph.py
            if hasattr(msg, "tool_calls") and msg.tool_calls and msg.tool_calls[0]['name'] == "download_website_text":
                yield "Downloading website text..."
            # if msg.additional_kwargs['tool_calls'] and msg.additional_kwargs['tool_calls'][0]== "download_website_text":
            print("output: ", msg, metadata)
            # assistant_node is the name we defined in the langraph graph
            if metadata['langgraph_node'] == "assistant_node" and msg.content:
                output += msg.content
                yield output
    except Exception:
        logger.exception("Exception occurred")
        user_error_message = "There was an error processing your request. Please try again."
        yield user_error_message  # , input_graph_state

def clear():
    return GraphProcessingState(), uuid4()

if __name__ == "__main__":
    load_dotenv()
    setup_logging()
    logger.info("Starting the interface")
    with gr.Blocks(title="Langgraph Template", fill_height=True, css="footer {visibility: hidden}") as app:
        gradio_graph_state = gr.State(
            value=GraphProcessingState
        )
        uuid_state = gr.State(
            uuid4
        )
        chatbot = gr.Chatbot(
            # avatar_images=(None, "assets/ai-avatar.png"),
            type="messages",
            # placeholder=WELCOME_MESSAGE,/
            scale=1,
        )
        chatbot.clear(fn=clear, outputs=[gradio_graph_state, uuid_state])
        chat_interface = gr.ChatInterface(
            chatbot=chatbot,
            fn=chat_fn,
            additional_inputs=[
                gradio_graph_state,
                uuid_state,
            ],
            additional_outputs=[
                # gradio_graph_state
            ],
            type="messages",
            multimodal=False,
        )

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        # favicon_path="assets/favicon.ico"
    )
