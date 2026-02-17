from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Event:
    name: str
    timestamp: float = field(default_factory=time.time)
    data: Optional[dict] = None


# --- Convenience constructors ---

def wake_word_detected(score: float = 0.0) -> Event:
    return Event("wake_word", data={"score": score})

def ready() -> Event:
    return Event("ready")

def voice_detected() -> Event:
    return Event("voice_detected")

def silence_detected(duration: float = 0.0) -> Event:
    return Event("silence", data={"duration": duration})

def timeout() -> Event:
    return Event("timeout")

def transcription_ready(text: str, language: str = "en") -> Event:
    return Event("transcription_ready", data={"text": text, "language": language})

def response_ready(text: str) -> Event:
    return Event("response_ready", data={"text": text})

def sentence_ready(text: str, is_last: bool = False) -> Event:
    return Event("sentence_ready", data={"text": text, "is_last": is_last})

def audio_finished() -> Event:
    return Event("done")

def error_occurred(message: str, exception: Optional[Exception] = None) -> Event:
    return Event("error", data={"message": message, "exception": str(exception) if exception else None})

def recover() -> Event:
    return Event("recover")
