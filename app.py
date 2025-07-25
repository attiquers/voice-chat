# app.py
import streamlit as st
import os
import logging
import numpy as np

# --- Import TTS Model ---
from kokoro_tts import KokoroTTS # Assuming kokoro_tts.py is in the same directory

# --- LangChain imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Iterator, List, Dict, Any

# --- LLM Specific Imports ---

# OPTION 1: For Ollama (Local)
# Make sure your Ollama server is running locally and the model is pulled
# import ollama

# OPTION 2: For Google Gemini (API)
# You will need to set GOOGLE_API_KEY in your .streamlit/secrets.toml (local)
# or Streamlit Cloud Secrets (deployment)
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

# --- LLM Initialization Logic ---
llm_stream_instance = None
selected_llm_type = "" # To store which LLM is active for display

# --- OPTION 1: Ollama LLM (Local) ---
# Uncomment the following block to use Ollama
"""
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest") # Default to llama3.2:latest or read from env
selected_llm_type = f"Ollama ({OLLAMA_MODEL})"

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
try:
    ollama_llm_stream_instance = OllamaStreamChatModel(model_name=OLLAMA_MODEL)
    llm_stream_instance = ollama_llm_stream_instance # Assign to the common variable
    logger.info(f"Ollama LLM initialized successfully with model: {OLLAMA_MODEL}.")
except Exception as e:
    logger.error(f"Failed to initialize Ollama LLM: {e}", exc_info=True)
    st.error(f"Error initializing Ollama LLM: {e}. Is Ollama server running?")
    llm_stream_instance = None
"""

# --- OPTION 2: Google Gemini LLM (API) ---
# Uncomment the following block to use Gemini
# Ensure GOOGLE_API_KEY is set in your .streamlit/secrets.toml or Streamlit Cloud Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
selected_llm_type = "Google Gemini (API)"

@st.cache_resource
def initialize_gemini_llm(api_key: str):
    if not api_key:
        logger.error("GOOGLE_API_KEY is not set. Gemini LLM cannot be initialized.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Using gemini-1.5-flash for broader access
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
# If you uncommented Ollama block, make sure to comment this line if you want to use Ollama primarily
# llm_stream_instance = initialize_gemini_llm(GOOGLE_API_KEY)


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

# --- MODIFIED: Process the full text response for audio generation ---
def _process_full_text_for_audio(full_text_response: str) -> Iterator[np.ndarray]:
    """
    Processes a complete text response, splits it into sentences/lines,
    synthesizes audio for each, and yields the audio chunks.
    """
    if not full_text_response.strip():
        return

    # A simple sentence tokenizer. You might use NLTK for more robust tokenization
    # if complex sentence structures are common.
    # For now, splitting by common sentence delimiters or newlines
    
    # Split by newlines first, then try to split lines into sentences
    lines = full_text_response.split('\n')
    sentences = []
    current_sentence_part = ""
    for line in lines:
        if not line.strip(): # Treat empty lines as sentence breaks
            if current_sentence_part.strip():
                sentences.append(current_sentence_part.strip())
            current_sentence_part = ""
            continue

        # Simple split by punctuation, keeping the punctuation
        for char in line:
            current_sentence_part += char
            if char in ['.', '!', '?']:
                if current_sentence_part.strip():
                    sentences.append(current_sentence_part.strip())
                current_sentence_part = ""
        
        # If line doesn't end with punctuation, append to current_sentence_part
        # This prevents breaking mid-sentence across newlines, but respects explicit newlines
        if current_sentence_part.strip() and not any(current_sentence_part.endswith(d) for d in ['.', '!', '?']):
            current_sentence_part += " " # Add space if continuing
    
    # Add any remaining part as a final sentence
    if current_sentence_part.strip():
        sentences.append(current_sentence_part.strip())

    logger.info(f"Full response parsed into {len(sentences)} audio chunks.")
    for sentence in sentences:
        if sentence:
            logger.info(f"Synthesizing sentence: '{sentence}'")
            audio_chunk = _synthesize_text_chunk(sentence)
            if audio_chunk.size > 0:
                yield audio_chunk

# Define the main LangChain chain
# This chain now only generates the full text response, it doesn't process for audio directly
main_chain = None
if llm_stream_instance: # Only build the chain if LLM was initialized successfully
    main_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in x["chat_history"]]
        )
        | prompt_template
        # The next line depends on which LLM is active:
        # If using Ollama: | RunnableLambda(llm_stream_instance.stream)
        # If using Gemini:  | llm_stream_instance
        | (RunnableLambda(llm_stream_instance.stream) if selected_llm_type.startswith("Ollama") else llm_stream_instance)
        | StrOutputParser() # This will combine all streamed tokens into one final string
    )

# --- Streamlit App ---

st.set_page_config(page_title="Streaming Text-to-Audio Chat", layout="wide")

st.markdown("# 💬➡️🔊 Streaming Chat (LLM + Kokoro TTS)")
st.markdown("LLM generates full response, then audio plays line-by-line!")

# Display status of TTS and LLM initialization
if kokoro_tts is None:
    st.warning("**⚠️ Warning: Kokoro TTS failed to initialize. Audio output will not be available.**")
else:
    st.success("Kokoro TTS initialized. Audio streaming enabled.")

if llm_stream_instance is None:
    st.error(f"**❌ Error: {selected_llm_type.split(' ')[0]} LLM failed to initialize.** Please check your setup (Ollama server/model or API key).")
else:
    st.success(f"{selected_llm_type} LLM initialized and ready.")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    # No need to reset accumulated_audio in session_state unless you store it for replay
    # The audio_placeholder will be updated with new audio each time.
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
        if msg["role"] != "user" or msg["content"] != user_input
    ]

    chain_input = {
        "input": user_input,
        "chat_history": langchain_chat_history_for_input
    }

    full_llm_response_text = ""
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # For streaming text response
        
        if main_chain is None:
            full_llm_response_text = f"LLM is not initialized. Please ensure {selected_llm_type.split(' ')[0]} setup is correct."
            message_placeholder.markdown(full_llm_response_text)
        else:
            try:
                # --- Step 1: Stream and accumulate the full text response from LLM ---
                for chunk in main_chain.stream(chain_input):
                    full_llm_response_text += chunk
                    message_placeholder.markdown(full_llm_response_text + "▌") # Show typing effect

                message_placeholder.markdown(full_llm_response_text) # Remove blinking cursor

                # Add the full response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_llm_response_text})
                
                # --- Step 2: Process the full text response for audio line-by-line ---
                if kokoro_tts:
                    accumulated_audio_for_playback = INITIAL_SILENCE.copy()
                    for audio_chunk in _process_full_text_for_audio(full_llm_response_text):
                        if audio_chunk.size > 0:
                            accumulated_audio_for_playback = np.concatenate((accumulated_audio_for_playback, audio_chunk), axis=None)
                            audio_placeholder.audio(accumulated_audio_for_playback, sample_rate=SAMPLE_RATE, format='audio/wav', autoplay=True)
                            logger.debug(f"Playing audio chunk. Total audio samples: {accumulated_audio_for_playback.shape[0]}")
                    logger.info("Finished playing all audio chunks for the response.")
                else:
                    st.warning("Audio synthesis skipped: Kokoro TTS not initialized.")

            except Exception as e:
                logger.error(f"Error during LLM or audio processing: {e}", exc_info=True)
                error_message = f"\n\nError: {str(e)}"
                full_llm_response_text += error_message
                message_placeholder.markdown(full_llm_response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": full_llm_response_text})