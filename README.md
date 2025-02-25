A template for chatbot streaming from langgraph with gradio

The template features history for the LLM, a couple tool calls, streaming output, and follow-up question buttons.

![image](https://github.com/user-attachments/assets/d98bd033-128e-427a-9d8e-a79eabeb338f)


# Install

    pip install uv

Add your openai api key to `.env`
Optionally add tavily key for websearch

# Run

    uv run app.py

# Chat with tabs

`app_with_tabs.py` is a version with tabs for multiple chat windows and persistence

You can run it with

    uv run app_with_tabs.py

![image](https://github.com/user-attachments/assets/1b8222a1-f63b-470d-b5db-d2ead49054f0)


# License

This code is published under the MIT License
