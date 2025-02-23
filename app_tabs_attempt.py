#!/usr/bin/env python

from uuid import uuid4
import logging
import logging.config
import json

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import RunnableConfig
from pydantic import BaseModel, Field

load_dotenv()

from graph import GraphProcessingState, graph, model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FOLLOWUP_QUESTION_NUMBER = 3

def setup_logging():
    pass
    # with open("logging-config.json") as fh:
    #     config = json.load(fh)
    # logging.config.dictConfig(config)

async def chat_fn(message, history, input_graph_state, uuid):
    try:
        input_graph_state["user_input"] = message
        input_graph_state["history"] = history
        config = RunnableConfig()
        config["configurable"] = {}
        config["configurable"]["thread_id"] = uuid

        output = ""
        async for msg, metadata in graph.astream(
                    dict(input_graph_state),
                    config=config,
                    stream_mode="messages",
                ):
            # download_website_text is the name of the function defined in graph.py
            if hasattr(msg, "tool_calls") and msg.tool_calls and msg.tool_calls[0]['name'] == "download_website_text":
                # yield {"role": "assistant", "content": "Downloading website text..."}
                yield "Downloading website text...", gr.skip(), False
            # if msg.additional_kwargs['tool_calls'] and msg.additional_kwargs['tool_calls'][0]== "download_website_text":
            # print("output: ", msg, metadata)
            # assistant_node is the name we defined in the langraph graph
            if metadata['langgraph_node'] == "assistant_node" and msg.content:
                output += msg.content
                yield output, gr.skip(), False
        # Trigger for asking follow up questions
        final_state = graph.get_state(config).values
        yield output, final_state, True
    except Exception:
        logger.exception("Exception occurred")
        user_error_message = "There was an error processing your request. Please try again."
        yield user_error_message, gr.skip(), False

def clear():
    return GraphProcessingState(), uuid4()

class FollowupQuestions(BaseModel):
    questions: list[str]

async def change_buttons(end_of_chat_response, messages):
    if not end_of_chat_response or not messages:
        return [gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)]
    if messages[-1]["role"] == "assistant":
        follow_up_questions: FollowupQuestions = await model.with_structured_output(FollowupQuestions).ainvoke([
            ("system", f"suggest {FOLLOWUP_QUESTION_NUMBER} followup questions"),
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

def summarize_chat(end_of_chat_response, messages, sidebar_summaries, uuid):
    if not end_of_chat_response:
        return gr.skip()
    if not messages:
        return gr.skip()
    if messages[-1]["role"] != "assistant":
        return gr.skip()
    if isinstance(sidebar_summaries, type(lambda x: x)):
        return gr.skip()
    # TODO make async
    if uuid in sidebar_summaries:
        return
    summary_response = model.invoke([
        ("system", "summarize this chat in 7 tokens or less. Refrain from using periods"),
        *messages,
    ])
    if uuid not in sidebar_summaries:
        sidebar_summaries[uuid] = summary_response.content
    return sidebar_summaries

def new_tab(uuid, gradio_graph, messages, tabs):
    new_uuid = uuid4()
    new_graph = {}
    tabs[uuid] = {
        "graph": gradio_graph,
        "messages": messages,
    }
    suggestion_buttons = []
    for _ in range(FOLLOWUP_QUESTION_NUMBER):
        suggestion_buttons.append(gr.Button(visible=False))
    chatbot = gr.Chatbot(
        type="messages",
        scale=1,
    )
    return new_uuid, new_graph, chatbot, tabs, *suggestion_buttons

def switch_tab(button_value, sidebar_summaries, tabs, gradio_graph, uuid, messages):
    # I don't know of another way to lookup uuid other than
    # by the button value
    tabs[uuid] = {
        "graph": gradio_graph,
        "messages": messages,
    }
    selected_uuid = None
    for uuid, summary in sidebar_summaries.items():
        if summary == button_value:
            selected_uuid = uuid
            break

    if not selected_uuid:
        logger.error("Could not find the selected tab for button {button_value}")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if selected_uuid not in tabs:
        logger.error(f"Could not find the selected tab in tabs_state {selected_uuid}")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    selected_tab_state = tabs[selected_uuid]
    selected_graph = selected_tab_state["graph"]
    selected_messages = selected_tab_state["messages"]
    return selected_graph, selected_uuid, selected_messages, tabs

CSS = """
footer {visibility: hidden}
.followup-question-button {font-size: 12px }
.chat-tab {
    font-size: 12px;
    padding-inline: 0;
}
.chat-tab.active {
    background-color: #654343;
}
#new-chat-button { background-color: #0f0f11; color: white; }
"""

if __name__ == "__main__":
    setup_logging()
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
        sidebar_names_state = gr.State(
            lambda: dict()
        )
        tabs_state = gr.State(
            lambda: dict()
        )
        chatbot = gr.Chatbot(
            # avatar_images=(None, "assets/ai-avatar.png"),
            type="messages",
            # placeholder=WELCOME_MESSAGE,/
            scale=1,
        )
        with gr.Sidebar() as sidebar:
            @gr.render(inputs=[end_of_chat_response_state, sidebar_names_state, uuid_state, chatbot])
            def render_chats(end_of_chat_response, current_chats, active_uuid, messages):
                # I suspect current_chats being a function is a bug in the gradio library
                if isinstance(current_chats, type(lambda x: x)):
                    return
                if active_uuid not in current_chats:
                    gr.Button("Current Chat", elem_id=f"chat-{active_uuid}-button", elem_classes=["chat-tab", "active"])
                for chat_uuid, summary in reversed(current_chats.items()):
                    # chat_tabs.append(
                    elem_classes = ["chat-tab"]
                    if chat_uuid == active_uuid:
                        elem_classes.append("active")
                    chat_tab_button = gr.Button(summary, elem_id=f"chat-{chat_uuid}-button", elem_classes=elem_classes)
                    chat_tab_button.click(fn=switch_tab, inputs=[chat_tab_button, sidebar_names_state, tabs_state, gradio_graph_state, uuid_state, chatbot], outputs=[gradio_graph_state, uuid_state, chatbot, tabs_state])
                    # )
                # return chat_tabs, current_chats
            new_chat_button = gr.Button("New Chat", elem_id="new-chat-button")
        chatbot.clear(fn=clear, outputs=[gradio_graph_state, uuid_state])
        with gr.Row():
            followup_question_buttons = []
            for i in range(FOLLOWUP_QUESTION_NUMBER):
                btn = gr.Button(f"Button {i+1}", visible=False, elem_classes="followup-question-button")
                followup_question_buttons.append(btn)

        new_chat_button.click(new_tab, inputs=[uuid_state, gradio_graph_state, chatbot, tabs_state], outputs=[uuid_state, gradio_graph_state, chatbot, tabs_state, *followup_question_buttons])

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

        chatbot.change(fn=change_buttons, inputs=[end_of_chat_response_state, chatbot], outputs=followup_question_buttons, trigger_mode="once")
        chatbot.change(fn=summarize_chat, inputs=[end_of_chat_response_state, chatbot, sidebar_names_state, uuid_state], outputs=[sidebar_names_state], trigger_mode="once")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        # favicon_path="assets/favicon.ico"
    )
