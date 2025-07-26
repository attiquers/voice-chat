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

# Init modules
if "stt" not in st.session_state:
    st.session_state.stt = FastWhisperSTT()

if "tts" not in st.session_state:
    st.session_state.tts = KokoroTTS()

# Init state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (user, ai) tuples

if "transcription_done" not in st.session_state:
    st.session_state.transcription_done = False

if "response_done" not in st.session_state:
    st.session_state.response_done = False

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# Layout
left, right = st.columns([1, 2])

# LEFT PANEL: Chat history
with left:
    st.header("🧠 Chat History")
    for user, ai in st.session_state.chat_history:
        st.markdown(f"**🧍 You:** {user}")
        st.markdown(f"**🤖 AI:** {ai}")

# RIGHT PANEL: Audio input and processing
with right:
    st.header("🎤 Record your message")
    audio_file = st.audio_input("Click and speak", key="audio_input")

    if audio_file:
        current_audio_hash = len(audio_file.getvalue())

        # New audio detected
        if current_audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = current_audio_hash
            st.session_state.audio_processed = False
            st.session_state.transcription_done = False
            st.session_state.response_done = False

        if not st.session_state.audio_processed:
            st.session_state.audio_processed = True

            with st.spinner("🔍 Transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                transcript = st.session_state.stt.transcribe(tmp_path)
                os.remove(tmp_path)

                # Add user message with placeholder AI message
                st.session_state.chat_history.append((transcript, "…"))
                st.session_state.transcription_done = True
                st.rerun()

    # LLM generation after transcription
    if st.session_state.transcription_done and not st.session_state.response_done:
        user_text = st.session_state.chat_history[-1][0]
        previous_context = st.session_state.chat_history[:-1]
        context = "\n".join([f"You: {u}\nAI: {a}" for u, a in previous_context])
        prompt = f"{context}\nYou: {user_text}" if context else user_text

        with st.spinner("💬 Generating AI response..."):
            response = get_response(prompt)
            st.session_state.chat_history[-1] = (user_text, response)
            st.session_state.response_done = True
            st.rerun()

    # TTS playback after response
    elif st.session_state.response_done:
        ai_response = st.session_state.chat_history[-1][1]
        lines = [line.strip() for line in ai_response.split(".") if line.strip()]
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
                    st.markdown(audio_html, unsafe_allow_html=True)

                os.remove(wav_path)
