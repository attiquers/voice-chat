import os
import streamlit as st
from dotenv import load_dotenv
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import wave
import json
from openai import OpenAI

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL")

# Set OpenAI key for OpenRouter
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# Initialize Vosk
vosk_model = Model("models/vosk")

def transcribe(audio_path):
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())

    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result += json.loads(rec.Result()).get("text", "")
    result += json.loads(rec.FinalResult()).get("text", "")
    return result.strip()

def convert_to_vosk_compatible(src_path, dst_path):
    # Converts to mono 16kHz WAV using pydub
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(dst_path, format="wav")

def generate_reply(prompt):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

st.title("🎙️ Streamlit Voice Chat with Vosk + OpenRouter")

audio_bytes = st.audio_input("Record something", format="audio/wav")
if audio_bytes is not None:
    with st.spinner("Processing audio..."):
        with open("input.wav", "wb") as f:
            f.write(audio_bytes.getvalue())

        convert_to_vosk_compatible("input.wav", "converted.wav")
        transcript = transcribe("converted.wav")
        st.markdown(f"**You said:** `{transcript}`")

        if transcript:
            reply = generate_reply(transcript)
            st.markdown(f"**Assistant:** {reply}")
