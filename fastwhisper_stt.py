# fastwhisper_stt.py
import os
import wave
import json
from vosk import Model, KaldiRecognizer

class VoskSTT:
    def __init__(self, model_path="models/vosk"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at {model_path}. "
                f"Make sure it’s downloaded and unzipped correctly."
            )
        self.model = Model(model_path)

    def transcribe(self, wav_path):
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(self.model, wf.getframerate())

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))

        return " ".join([r.get("text", "") for r in results])
