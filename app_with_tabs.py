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
USER_INPUT_MAX_LENGTH = 10000  # Characters

# We need the same secret for data persistance
# If you store sensitive data, you should store your secret in .env
BROWSER_STORAGE_SECRET = "itsnosecret"

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
        config = RunnableConfig(
            recursion_limit=10,
            run_name="user_chat",
            configurable={"thread_id": uuid}
        )

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
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for msg_tool_call in msg.tool_calls:
                        tool_name: str = msg_tool_call['name']
                        # download_website_text is the name of the function defined in graph.py
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
    This function gets called a lot due to the asynchronous nature of streaming

    Only populate followup questions if streaming has completed and the message is coming from the assistant
    """
    if not end_of_chat_response or not messages:
        return *[gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)], False
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
        return *buttons, False
    else:
        return *[gr.skip() for _ in range(FOLLOWUP_QUESTION_NUMBER)], False

async def summarize_chat(end_of_chat_response, messages, sidebar_summaries, uuid):
    if not end_of_chat_response:
        return gr.skip()
    if not messages:
        return gr.skip()
    if messages[-1]["role"] != "assistant":
        return gr.skip()
    if isinstance(sidebar_summaries, type(lambda x: x)):
        return gr.skip()
    if sidebar_summaries is None:
        return gr.skip()
    if uuid in sidebar_summaries:
        return gr.skip()
    summary_response = await model.ainvoke([
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
    new_messages = {}
    return new_uuid, new_graph, new_messages, tabs, *suggestion_buttons

def switch_tab(selected_uuid, tabs, gradio_graph, uuid, messages):
    # I don't know of another way to lookup uuid other than
    # by the button value

    # Save current state
    tabs[uuid] = {
        "graph": gradio_graph,
        "messages": messages,
    }

    if selected_uuid not in tabs:
        logger.error(f"Could not find the selected tab in offloaded_tabs_data_storage {selected_uuid}")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    selected_tab_state = tabs[selected_uuid]
    selected_graph = selected_tab_state["graph"]
    selected_messages = selected_tab_state["messages"]
    suggestion_buttons = []
    for _ in range(FOLLOWUP_QUESTION_NUMBER):
        suggestion_buttons.append(gr.Button(visible=False))
    return selected_graph, selected_uuid, selected_messages, tabs, *suggestion_buttons

def delete_tab(current_chat_uuid, selected_uuid, sidebar_summaries, tabs):
    output_messages = gr.skip()
    if current_chat_uuid == selected_uuid:
        output_messages = dict()
    if selected_uuid in tabs:
        del tabs[selected_uuid]
    del sidebar_summaries[selected_uuid]
    return sidebar_summaries, tabs, output_messages

def submit_edit_tab(selected_uuid, sidebar_summaries, text):
    sidebar_summaries[selected_uuid] = text
    return sidebar_summaries, ""

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

.tab-button-control {
    min-width: 0;
    padding-left: 0;
    padding-right: 0;
}
"""

# We set the ChatInterface textbox id to chat-textbox for this to work
TRIGGER_CHATINTERFACE_BUTTON = """
function triggerChatButtonClick() {

  // Find the div with id "chat-textbox"
  const chatTextbox = document.getElementById("chat-textbox");

  if (!chatTextbox) {
    console.error("Error: Could not find element with id 'chat-textbox'");
    return;
  }

  // Find the button that is a descendant of the div
  const button = chatTextbox.querySelector("button");

  if (!button) {
    console.error("Error: No button found inside the chat-textbox element");
    return;
  }

  // Trigger the click event
  button.click();
}"""

if __name__ == "__main__":
    logger.info("Starting the interface")
    with gr.Blocks(title="Langgraph Template", fill_height=True, css=CSS) as app:
        current_uuid_state = gr.BrowserState(
            uuid4,
            storage_key="current_uuid_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        current_langgraph_state = gr.BrowserState(
            dict(),
            storage_key="current_langgraph_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        end_of_assistant_response_state = gr.State(
            bool(),
        )
        # [uuid] -> summary of chat
        sidebar_names_state = gr.BrowserState(
            dict(),
            storage_key="sidebar_names_state",
            secret=BROWSER_STORAGE_SECRET,
        )
        # [uuid] -> {"graph": gradio_graph, "messages": messages}
        offloaded_tabs_data_storage = gr.BrowserState(
            dict(),
            storage_key="offloaded_tabs_data_storage",
            secret=BROWSER_STORAGE_SECRET,
        )

        chatbot_message_storage = gr.BrowserState(
            [],
            storage_key="chatbot_message_storage",
            secret=BROWSER_STORAGE_SECRET,
        )
        chatbot = gr.Chatbot(
            type="messages",
            scale=1,
        )
        tab_edit_uuid_state = gr.State(
            str()
        )
        with gr.Sidebar() as sidebar:
            @gr.render(inputs=[tab_edit_uuid_state, end_of_assistant_response_state, sidebar_names_state, current_uuid_state, chatbot])
            def render_chats(tab_uuid_edit, end_of_chat_response, current_chats, active_uuid, messages):
                if active_uuid not in current_chats:
                    gr.Button("Current Chat", elem_id=f"chat-{active_uuid}-button", elem_classes=["chat-tab", "active"])
                for chat_uuid, summary in reversed(current_chats.items()):
                    # chat_tabs.append(
                    elem_classes = ["chat-tab"]
                    if chat_uuid == active_uuid:
                        elem_classes.append("active")
                    button_uuid_state = gr.State(chat_uuid)
                    with gr.Row():
                        clear_tab_button = gr.Button("🗑", scale=0, elem_classes=["tab-button-control"])
                        clear_tab_button.click(fn=delete_tab, inputs=[current_uuid_state, button_uuid_state, sidebar_names_state, offloaded_tabs_data_storage], outputs=[sidebar_names_state, offloaded_tabs_data_storage, chat_interface.chatbot_value])
                        if chat_uuid != tab_uuid_edit:
                            set_edit_tab_button = gr.Button("✎", scale=0, elem_classes=["tab-button-control"])
                            set_edit_tab_button.click(fn=lambda x: x, inputs=[button_uuid_state], outputs=[tab_edit_uuid_state])
                            chat_tab_button = gr.Button(summary, elem_id=f"chat-{chat_uuid}-button", elem_classes=elem_classes, scale=2)
                            chat_tab_button.click(fn=switch_tab, inputs=[button_uuid_state, offloaded_tabs_data_storage, current_langgraph_state, current_uuid_state, chatbot], outputs=[current_langgraph_state, current_uuid_state, chat_interface.chatbot_value, offloaded_tabs_data_storage, *followup_question_buttons])
                        else:
                            chat_tab_text = gr.Textbox(summary, elem_id=f"chat-{chat_uuid}-button", scale=2, interactive=True, show_label=False)
                            chat_tab_text.submit(fn=submit_edit_tab, inputs=[button_uuid_state, sidebar_names_state, chat_tab_text], outputs=[sidebar_names_state, tab_edit_uuid_state])
                    # )
                # return chat_tabs, current_chats
            new_chat_button = gr.Button("New Chat", elem_id="new-chat-button")
        chatbot.clear(fn=clear, outputs=[current_langgraph_state, current_uuid_state])
        with gr.Row():
            followup_question_buttons = []
            for i in range(FOLLOWUP_QUESTION_NUMBER):
                btn = gr.Button(f"Button {i+1}", visible=False, elem_classes="followup-question-button")
                followup_question_buttons.append(btn)


        multimodal = False
        textbox_component = (
            gr.MultimodalTextbox if multimodal else gr.Textbox
        )
        with gr.Column():
            textbox = textbox_component(
                show_label=False,
                label="Message",
                placeholder="Type a message...",
                scale=7,
                autofocus=True,
                submit_btn=True,
                stop_btn=True,
                elem_id="chat-textbox",
                lines=1,
            )
        chat_interface = gr.ChatInterface(
            chatbot=chatbot,
            fn=chat_fn,
            additional_inputs=[
                current_langgraph_state,
                current_uuid_state,
            ],
            additional_outputs=[
                current_langgraph_state,
                end_of_assistant_response_state
            ],
            type="messages",
            multimodal=multimodal,
            textbox=textbox,
        )

        new_chat_button.click(new_tab, inputs=[current_uuid_state, current_langgraph_state, chatbot, offloaded_tabs_data_storage], outputs=[current_uuid_state, current_langgraph_state, chat_interface.chatbot_value, offloaded_tabs_data_storage, *followup_question_buttons])

        def click_followup_button(btn):
            buttons = [gr.Button(visible=False) for _ in range(len(followup_question_buttons))]
            return btn, *buttons
        for btn in followup_question_buttons:
            btn.click(fn=click_followup_button, inputs=[btn], outputs=[chat_interface.textbox, *followup_question_buttons]).success(lambda: None, js=TRIGGER_CHATINTERFACE_BUTTON)

        chatbot.change(fn=populate_followup_questions, inputs=[end_of_assistant_response_state, chatbot], outputs=[*followup_question_buttons, end_of_assistant_response_state], trigger_mode="once")
        chatbot.change(fn=summarize_chat, inputs=[end_of_assistant_response_state, chatbot, sidebar_names_state, current_uuid_state], outputs=[sidebar_names_state], trigger_mode="once")
        chatbot.change(fn=lambda x: x, inputs=[chatbot], outputs=[chatbot_message_storage], trigger_mode="once")

        @app.load(inputs=[chatbot_message_storage], outputs=[chat_interface.chatbot_value])
        def load_messages(messages):
            return messages

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        # favicon_path="assets/favicon.ico"
    )
