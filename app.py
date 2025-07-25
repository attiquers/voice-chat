# app.py
import streamlit as st
import os
import logging
import numpy as np
# No direct 'import ollama' as we are using Gemini API now
from kokoro_tts import KokoroTTS # Assuming kokoro_tts.py is in the same directory

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator, List, Dict, Any

# Import Google Gemini integration for LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
SAMPLE_RATE = 24000
INITIAL_SILENCE_DURATION = 0.1 # seconds
INITIAL_SILENCE_SAMPLES = int(SAMPLE_RATE * INITIAL_SILENCE_DURATION)
INITIAL_SILENCE = np.zeros(INITIAL_SILENCE_SAMPLES, dtype=np.float32)

# --- Initialize Kokoro TTS (cached for performance) ---
@st.cache_resource
def initialize_kokoro_tts():
    logger.info("Initializing Kokoro TTS...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "models")
        model_path = os.path.join(model_dir, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(model_dir, "voices-v1.0.bin")

        os.makedirs(model_dir, exist_ok=True)

        logger.info(f"Checking for model file: {os.path.abspath(model_path)} - Exists: {os.path.exists(model_path)}")
        logger.info(f"Checking for voices file: {os.path.abspath(voices_path)} - Exists: {os.path.exists(voices_path)}")

        tts = KokoroTTS(model_path, voices_path)
        logger.info("Kokoro TTS initialized successfully.")
        return tts
    except Exception as e:
        logger.error(f"Failed to initialize Kokoro TTS: {e}", exc_info=True)
        return None

kokoro_tts = initialize_kokoro_tts()

# --- Initialize Gemini LLM (cached for performance) ---
# GOOGLE_API_KEY will be loaded from .streamlit/secrets.toml (local) or Streamlit Cloud secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def initialize_gemini_llm(api_key: str):
    if not api_key:
        logger.error("GOOGLE_API_KEY is not set. Gemini LLM cannot be initialized.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # You can choose other models like "gemini-1.5-pro", "gemini-1.5-flash"
            temperature=0.7,
            streaming=True,
            google_api_key=api_key
        )
        logger.info("Gemini LLM initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {e}", exc_info=True)
        st.error("Error initializing Gemini LLM. Please check your GOOGLE_API_KEY and model access.")
        return None

llm_stream_instance = initialize_gemini_llm(GOOGLE_API_KEY)


# --- LangChain Components ---

# Prompt Template: LangChain handles message roles and history
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Answer with each point or sentence on a separate line."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Helper function to synthesize a single text chunk with KokoroTTS
def _synthesize_text_chunk(text_chunk: str) -> np.ndarray:
    if not text_chunk.strip():
        return np.array([])
    if kokoro_tts:
        try:
            audio = kokoro_tts.synthesize(text_chunk.strip())
            return audio if audio is not None else np.array([])
        except Exception as e:
            logger.error(f"Synthesis error for '{text_chunk.strip()}': {e}", exc_info=True)
            return np.array([])
    else:
        logger.warning("Kokoro TTS not initialized. Cannot synthesize audio.")
        return np.array([])

def _process_text_for_audio(text_stream: Iterator[str]) -> Iterator[Dict[str, Any]]:
    """
    Processes a stream of text tokens, identifies completed sentences/lines, and synthesizes audio.
    Yields dictionaries containing either 'text_token' or 'audio_chunk' for downstream processing.
    """
    accumulated_text = ""
    # Define common sentence-ending punctuation
    sentence_delimiters = ['.', '!', '?', '\n']

    for token in text_stream:
        accumulated_text += token
        
        yield {"type": "text_token", "content": token}

        # Check if the current token completes a sentence or line
        if any(delimiter in token for delimiter in sentence_delimiters):
            # Split the accumulated text into potential sentences
            # This is a simple split; more advanced NLP could be used for better sentence boundary detection
            
            # Find the last delimiter to split only completed parts
            last_delimiter_idx = -1
            for delimiter in reversed(sentence_delimiters):
                if delimiter in accumulated_text:
                    idx = accumulated_text.rfind(delimiter)
                    if idx > last_delimiter_idx:
                        last_delimiter_idx = idx

            if last_delimiter_idx != -1:
                # Extract the completed part (including the delimiter)
                completed_part = accumulated_text[:last_delimiter_idx + 1]
                # Keep the remainder for the next accumulation
                accumulated_text = accumulated_text[last_delimiter_idx + 1:]

                line_text = completed_part.strip()
                if line_text:
                    logger.info(f"Processing completed chunk for audio: '{line_text}'")
                    audio_chunk = _synthesize_text_chunk(line_text)
                    if audio_chunk.size > 0:
                        yield {"type": "audio_chunk", "content": audio_chunk}

    # After the stream ends, process any remaining accumulated text
    if accumulated_text.strip():
        final_line_text = accumulated_text.strip()
        logger.info(f"Processing final incomplete chunk for audio: '{final_line_text}'")
        final_audio_chunk = _synthesize_text_chunk(final_line_text)
        if final_audio_chunk.size > 0:
            yield {"type": "audio_chunk", "content": final_audio_chunk}

# Define the main LangChain chain
main_chain = None
if llm_stream_instance: # Only build the chain if LLM was initialized successfully
    main_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in x["chat_history"]]
        )
        | prompt_template
        | llm_stream_instance # Direct invocation of the LangChain LLM
        | StrOutputParser()
        | RunnableLambda(_process_text_for_audio)
    )

# --- Streamlit App ---

st.set_page_config(page_title="Streaming Text-to-Audio Chat", layout="wide")

st.markdown("# 💬➡️🔊 Streaming Chat (LLM + Kokoro TTS)")
st.markdown("LLM responds line-by-line, audio plays continuously!")

# Display status of TTS and LLM initialization
if kokoro_tts is None:
    st.warning("**⚠️ Warning: Kokoro TTS failed to initialize. Audio output will not be available.**")
else:
    st.success("Kokoro TTS initialized. Audio streaming enabled.")

if llm_stream_instance is None:
    st.error("**❌ Error: Gemini LLM failed to initialize. Please check your `GOOGLE_API_KEY` in Streamlit Secrets.**")
else:
    st.success("Gemini LLM initialized and ready.")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "accumulated_audio" not in st.session_state:
    st.session_state.accumulated_audio = INITIAL_SILENCE.copy()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Placeholder for the audio player (will be updated dynamically)
audio_placeholder = st.empty()

# Input box for user message
user_input = st.chat_input("Type your message here...")

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.accumulated_audio = INITIAL_SILENCE.copy()

st.sidebar.button("🗑️ Clear Chat", on_click=clear_chat)

if user_input:
    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare chat history for LangChain input
    # The current user input is handled by the `input` field of the chain,
    # so we build chat_history with previous messages only.
    langchain_chat_history_for_input = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history
        if msg["role"] != "user" or msg["content"] != user_input # Exclude the current user input from history
    ]

    chain_input = {
        "input": user_input,
        "chat_history": langchain_chat_history_for_input
    }

    current_assistant_response_content = ""
    # Reset audio for the new response
    st.session_state.accumulated_audio = INITIAL_SILENCE.copy() 

    with st.chat_message("assistant"):
        # Create an empty container for the assistant's streaming text
        full_response_text = ""
        message_placeholder = st.empty() # This will hold the markdown content

        if main_chain is None:
            full_response_text = "LLM is not initialized. Please check API key setup."
            message_placeholder.markdown(full_response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
        else:
            try:
                # Stream from the LangChain main_chain
                for chunk_data in main_chain.stream(chain_input):
                    if chunk_data["type"] == "text_token":
                        full_response_text += chunk_data["content"]
                        # Update text with a blinking cursor effect
                        message_placeholder.markdown(full_response_text + "▌") 
                    elif chunk_data["type"] == "audio_chunk":
                        audio_segment = chunk_data["content"]
                        if audio_segment.size > 0:
                            st.session_state.accumulated_audio = np.concatenate((st.session_state.accumulated_audio, audio_segment), axis=None)
                            audio_placeholder.audio(st.session_state.accumulated_audio, sample_rate=SAMPLE_RATE, format='audio/wav')
                            logger.debug(f"Appended audio. New total audio samples: {st.session_state.accumulated_audio.shape[0]}")

                # After stream completes, finalize the text and audio updates
                message_placeholder.markdown(full_response_text) # Remove blinking cursor
                st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
                # Ensure final audio is played
                audio_placeholder.audio(st.session_state.accumulated_audio, sample_rate=SAMPLE_RATE, format='audio/wav')
                logger.info("Stream completed. Final audio played.")

            except Exception as e:
                logger.error(f"Error during Streamlit streaming: {e}", exc_info=True)
                full_response_text += f"\n\nError: {str(e)}"
                message_placeholder.markdown(full_response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})