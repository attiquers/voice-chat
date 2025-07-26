import streamlit as st
import tempfile
import os
import base64
import soundfile as sf

from fastwhisper_stt import FastWhisperSTT
from kokoro_tts import KokoroTTS
from llm_chat import get_response

st.set_page_config(layout="wide")
st.title("🎙️ Voice Assistant")

# Initialize modules
if "stt" not in st.session_state:
    st.session_state.stt = FastWhisperSTT()
if "tts" not in st.session_state:
    st.session_state.tts = KokoroTTS()

# Initialize state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcription_done" not in st.session_state:
    st.session_state.transcription_done = False
if "response_done" not in st.session_state:
    st.session_state.response_done = False
if "last_audio_blocks" not in st.session_state:
    st.session_state.last_audio_blocks = []
# New state variable to control the key of st.audio_input
# Changing the key forces Streamlit to re-render the widget, effectively resetting its state.
if "audio_input_key_counter" not in st.session_state:
    st.session_state.audio_input_key_counter = 0

# Layout
left, right = st.columns([1, 2])

# Left column: chat history
with left:
    st.header("🧠 Chat History")
    # Display chat history in reverse order to show latest at the top
    for user, ai in reversed(st.session_state.chat_history):
        st.markdown(f"**🧍 You:** {user}")
        st.markdown(f"**🤖 AI:** {ai}")

# Right column: audio input and processing
with right:
    st.header("🎤 Record your message")
    # Use a dynamic key for st.audio_input. When this key changes,
    # Streamlit considers it a new widget instance, resetting its internal state.
    audio_file = st.audio_input("Click and speak", key=f"audio_input_{st.session_state.audio_input_key_counter}")

    # This block will now execute ONLY when a new audio input is detected
    # because the 'key' mechanism ensures audio_file is None unless a new recording is made.
    if audio_file:
        # Clear previously displayed audio players from the UI
        for audio_block in st.session_state.last_audio_blocks:
            audio_block.empty()
        st.session_state.last_audio_blocks = []  # Clear the list itself for new audio blocks

        # Reset processing flags to allow a new transcription cycle to begin
        st.session_state.transcription_done = False
        st.session_state.response_done = False

        # Transcribe
        with st.spinner("🔍 Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            transcript = st.session_state.stt.transcribe(tmp_path)
            os.remove(tmp_path)
            st.session_state.chat_history.append((transcript, "…"))
            st.session_state.transcription_done = True
            # Increment the key counter. This forces st.audio_input to reset on the next rerun,
            # making audio_file None until a new recording is provided by the user.
            st.session_state.audio_input_key_counter += 1
            st.rerun() # Rerun the script to proceed to the next step (LLM response)

    # These subsequent `elif` blocks will correctly trigger after `st.rerun()`
    # because `transcription_done` will be True and `response_done` will be False,
    # or both True for TTS, allowing the multi-step process to continue.
    elif st.session_state.transcription_done and not st.session_state.response_done:
        user_text = st.session_state.chat_history[-1][0]
        # Construct full context for LLM
        full_context_parts = []
        for u, a in st.session_state.chat_history[:-1]:
            full_context_parts.append(f"You: {u}")
            full_context_parts.append(f"AI: {a}")
        full_context = "\n".join(full_context_parts)

        prompt = f"{full_context}\nYou: {user_text}" if full_context else user_text

        with st.spinner("💬 Generating AI response..."):
            response = get_response(prompt)
            # Update the last entry in chat history with the AI's response
            st.session_state.chat_history[-1] = (user_text, response)
            st.session_state.response_done = True
            st.rerun()

    elif st.session_state.response_done:
        ai_response = st.session_state.chat_history[-1][1]
        lines = [line.strip() for line in ai_response.split(".") if line.strip()]
        if lines: # Only show subheader if there's content to speak
            st.subheader("🔊 AI Speaking...")

        for i, line in enumerate(lines):
            with st.spinner(f"🗣️ Generating speech for sentence {i+1}..."):
                audio = st.session_state.tts.synthesize(line)
                if audio.size == 0:
                    continue

                wav_path = f"/tmp/tts_{i}.wav"
                sf.write(wav_path, audio, samplerate=24000)

                with open(wav_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    audio_html = f"""
                    <audio autoplay controls>
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    """
                    # Use st.empty() to create a placeholder that can be cleared later
                    audio_block = st.empty()
                    audio_block.markdown(audio_html, unsafe_allow_html=True)
                    st.session_state.last_audio_blocks.append(audio_block)

                os.remove(wav_path)
