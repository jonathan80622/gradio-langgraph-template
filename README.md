A template for chatbot streaming from langgraph with gradio

The template features history for the LLM, a couple tool calls, streaming output, and follow-up question buttons.

![image](https://github.com/user-attachments/assets/d98bd033-128e-427a-9d8e-a79eabeb338f)


# Install

    pip install uv
    uv sync
    cp .env.example .env

Add your openai api key to `.env`

# Run

    uv run app.py
