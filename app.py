import streamlit as st
import tempfile
import os
from vosk import Model, KaldiRecognizer
import wave
import json
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from kokoro_tts import KokoroTTS

# Paths
VOSK_MODEL_PATH = "models/vosk"
KOKORO_MODEL_PATH = "models/kokoro-v1.0.int8.onnx"
KOKORO_VOICE_PATH = "models/voices-v1.0.bin"

# Initialize models
stt_model = Model(VOSK_MODEL_PATH)
tts_engine = KokoroTTS(KOKORO_MODEL_PATH, KOKORO_VOICE_PATH)

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    model="mistralai/mistral-7b-instruct",  # or "openai/gpt-3.5-turbo", etc.
)

st.title("🗣️ Streamlit Voice Chat App (Offline STT, Kokoro TTS, OpenRouter LLM)")

# Audio input
audio_file = st.audio_input("🎙️ Record something", key="audio")

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    # Transcribe with Vosk
    wf = wave.open(temp_audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        st.error("Unsupported audio format. Please record in 16-bit mono WAV.")
    else:
        recognizer = KaldiRecognizer(stt_model, wf.getframerate())
        transcription = ""

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                transcription += res.get("text", "") + " "
        res = json.loads(recognizer.FinalResult())
        transcription += res.get("text", "")

        st.subheader("📝 Transcription")
        st.write(transcription)

        if transcription.strip():
            # LLM response
            response = llm.invoke([HumanMessage(content=transcription)])
            reply = response.content.strip()

            st.subheader("🤖 LLM Response")
            st.write(reply)

            # TTS synthesis
            audio_path = tts_engine.synthesize(reply)
            if audio_path:
                st.audio(audio_path, format="audio/wav")
            else:
                st.error("❌ TTS failed to generate audio.")
        else:
            st.warning("No transcription found.")
