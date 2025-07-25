# app.py
import streamlit as st
import os
import logging
import numpy as np
import ollama
from kokoro_tts import KokoroTTS # Assuming kokoro_tts.py is in the same directory

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator, List, Dict, Any

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
SAMPLE_RATE = 24000
INITIAL_SILENCE_DURATION = 0.1 # seconds
INITIAL_SILENCE_SAMPLES = int(SAMPLE_RATE * INITIAL_SILENCE_DURATION)
INITIAL_SILENCE = np.zeros(INITIAL_SILENCE_SAMPLES, dtype=np.float32)

# --- Initialize Models ---
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

# --- LangChain Components ---

# Custom Ollama LLM wrapper to integrate with LangChain's Runnable interface
class OllamaStreamChatModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def stream(self, prompt_value: Any) -> Iterator[str]:
        messages = prompt_value.to_messages()

        context_parts = []
        for msg in messages[:-1]:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
            
        current_user_message = ""
        if messages and isinstance(messages[-1], HumanMessage):
            current_user_message = messages[-1].content
            
        full_context = "\n".join(context_parts)
        
        system_instruction_prefix = ""
        if messages and isinstance(messages[0], BaseMessage) and hasattr(messages[0], 'type') and messages[0].type == 'system':
            system_instruction_prefix = messages[0].content + "\n"

        final_ollama_prompt = f"{system_instruction_prefix}{full_context}\nHuman: {current_user_message}\nAssistant:"
        
        logger.info(f"Ollama streaming with prompt preview: {final_ollama_prompt[:200]}...")
        try:
            stream = ollama.generate(model=self.model_name, prompt=final_ollama_prompt, stream=True)
            for chunk in stream:
                token = chunk.get("response", "")
                if token:
                    yield token
        except Exception as e:
            logger.error(f"Error in Ollama stream: {e}", exc_info=True)
            yield f"Error: {str(e)}"

# Instantiate your custom Ollama LLM wrapper
ollama_llm_stream_instance = OllamaStreamChatModel(model_name=OLLAMA_MODEL)

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
    accumulated_text = ""
    for token in text_stream:
        accumulated_text += token
        
        yield {"type": "text_token", "content": token}

        if '\n' in token:
            lines = accumulated_text.split('\n')
            completed_lines = lines[:-1]
            accumulated_text = lines[-1]

            for line in completed_lines:
                line_text = line.strip()
                if line_text:
                    logger.info(f"Processing completed line for audio: '{line_text}'")
                    audio_chunk = _synthesize_text_chunk(line_text)
                    if audio_chunk.size > 0:
                        yield {"type": "audio_chunk", "content": audio_chunk}

    if accumulated_text.strip():
        final_line_text = accumulated_text.strip()
        logger.info(f"Processing final incomplete line for audio: '{final_line_text}'")
        final_audio_chunk = _synthesize_text_chunk(final_line_text)
        if final_audio_chunk.size > 0:
            yield {"type": "audio_chunk", "content": final_audio_chunk}

# Define the main LangChain chain
main_chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in x["chat_history"]]
    )
    | prompt_template
    | RunnableLambda(ollama_llm_stream_instance.stream)
    | StrOutputParser()
    | RunnableLambda(_process_text_for_audio)
)

# --- Streamlit App ---

st.set_page_config(page_title="Streaming Text-to-Audio Chat", layout="wide")

st.markdown("# 💬➡️🔊 Streaming Chat (LLM + Kokoro TTS)")
st.markdown("LLM responds line-by-line, audio plays continuously!")

if kokoro_tts is None:
    st.warning("**⚠️ Warning: Kokoro TTS failed to initialize. Audio output will not be available.**")
else:
    st.success("Kokoro TTS initialized. Audio streaming enabled.")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "accumulated_audio" not in st.session_state:
    st.session_state.accumulated_audio = INITIAL_SILENCE.copy()

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Placeholders for streaming output
audio_placeholder = st.empty()
chat_message_placeholder = st.empty() # For updating the assistant's message in real-time

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
    langchain_chat_history_for_input = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history
        if msg["role"] != "user" or msg["content"] != user_input # Exclude the current user input from history for the chain's internal processing
    ]

    chain_input = {
        "input": user_input,
        "chat_history": langchain_chat_history_for_input
    }

    current_assistant_response_content = ""
    st.session_state.accumulated_audio = INITIAL_SILENCE.copy() # Reset audio for new response

    with st.chat_message("assistant"):
        # Create an empty container for the assistant's streaming text
        full_response_text = ""
        message_placeholder = st.empty()
        
        try:
            for chunk_data in main_chain.stream(chain_input):
                if chunk_data["type"] == "text_token":
                    full_response_text += chunk_data["content"]
                    message_placeholder.markdown(full_response_text + "▌") # Add blinking cursor effect
                elif chunk_data["type"] == "audio_chunk":
                    audio_segment = chunk_data["content"]
                    if audio_segment.size > 0:
                        st.session_state.accumulated_audio = np.concatenate((st.session_state.accumulated_audio, audio_segment), axis=None)
                        audio_placeholder.audio(st.session_state.accumulated_audio, sample_rate=SAMPLE_RATE, format='audio/wav')
                        logger.debug(f"Appended audio. New total audio samples: {st.session_state.accumulated_audio.shape[0]}")

            # After stream completes, update the final text and audio
            message_placeholder.markdown(full_response_text) # Remove blinking cursor
            st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})
            audio_placeholder.audio(st.session_state.accumulated_audio, sample_rate=SAMPLE_RATE, format='audio/wav')

        except Exception as e:
            logger.error(f"Error during Streamlit streaming: {e}", exc_info=True)
            full_response_text += f"\n\nError: {str(e)}"
            message_placeholder.markdown(full_response_text)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response_text})