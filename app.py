# app.py
import streamlit as st
import tempfile
import soundfile as sf

from kokoro_tts import KokoroTTS
from fastwhisper_stt import FastWhisperSTT

# === Initialize Models ===
tts = KokoroTTS()
stt_whisper = FastWhisperSTT()

# === Streamlit UI ===
st.title("🎙️ Speak to Kokoro (Offline Voice Chat)")

audio_input = st.audio_input("🎤 Record a message")
voice_id = st.selectbox("🗣️ Choose voice", tts.get_available_voices())

if audio_input is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_input.read())
        tmpfile_path = tmpfile.name

    st.audio(tmpfile_path, format="audio/wav")

    st.subheader("🧠 Transcription")
    text = stt_whisper.transcribe(tmpfile_path)
    st.write(text)

    st.subheader("🗣️ Kokoro Response")
    tts_audio = tts.synthesize(text, voice_id=voice_id)

    if tts_audio.size > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
            sf.write(out_f.name, tts_audio, samplerate=24000)
            st.audio(out_f.name, format="audio/wav")
