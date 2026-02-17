import logging
import wave
import struct
from typing import Optional, Generator

logger = logging.getLogger("nanobot.audio.capture")

# PyAudio may not be available in dev environment
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logger.warning("PyAudio not available — audio capture disabled")


class AudioCapture:
    def __init__(self, config: dict):
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.chunk_size = config.get("chunk_size", 480)
        self.device_index = config.get("device_index")
        self.pa: Optional[object] = None
        self.stream = None

        if HAS_PYAUDIO:
            self.pa = pyaudio.PyAudio()
            self._find_device()

    def _find_device(self):
        """Auto-detect USB audio input device if not specified."""
        if self.device_index is not None:
            return

        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            max_in = info.get("maxInputChannels", 0)
            if max_in > 0 and ("usb" in name or "pnp" in name or "respeaker" in name):
                self.device_index = i
                logger.info(f"Auto-detected input device [{i}]: {info['name']}")
                return

        logger.warning("No USB audio device found, using system default")

    def open_stream(self):
        """Open the audio capture stream."""
        if not HAS_PYAUDIO:
            raise RuntimeError("PyAudio not installed")

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        logger.debug("Audio capture stream opened")

    def read_chunk(self) -> bytes:
        """Read one chunk of audio data."""
        if self.stream is None:
            self.open_stream()
        return self.stream.read(self.chunk_size, exception_on_overflow=False)

    def read_chunks(self) -> Generator[bytes, None, None]:
        """Generator that yields audio chunks continuously."""
        if self.stream is None:
            self.open_stream()
        while True:
            yield self.read_chunk()

    def close(self):
        """Close stream and terminate PyAudio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.pa:
            self.pa.terminate()
            self.pa = None

    @staticmethod
    def save_wav(frames: list[bytes], path: str, sample_rate: int = 16000,
                 channels: int = 1, sample_width: int = 2):
        """Save raw audio frames to a WAV file."""
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        logger.debug(f"Saved {len(frames)} frames to {path}")
