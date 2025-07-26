# fastwhisper_stt.py
from faster_whisper import WhisperModel

class FastWhisperSTT:
    def __init__(self):
        self.model = WhisperModel(
            model_size_or_path="models/faster-whisper-tiny.en",
            compute_type="int8",  # optimized for CPU
            local_files_only=True
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)
