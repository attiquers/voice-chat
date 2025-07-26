from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import streamlit as st

def get_response(user_input: str) -> str:
    api_key = st.secrets["OPENROUTER_API_KEY"]

    # Use OpenRouter endpoint with Gemma
    llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model="google/gemma-3n-e4b-it:free",
    )

    # Prepend the assistant instruction directly into user prompt
    instruction = (
        "You are a voice assistant. Your response will be spoken aloud using TTS, write only text that can be read in normal conversation, "
        "so keep it friendly, conversational, and short. Use natural language.\n\n"
        f"User said: {user_input}\n\n"
        "Assistant reply:"
    )

    message = HumanMessage(content=instruction)

    return llm.invoke([message]).content
