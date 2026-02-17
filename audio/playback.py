import logging
import wave
from typing import Optional

logger = logging.getLogger("nanobot.audio.playback")

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


class AudioPlayback:
    def __init__(self, config: dict):
        self.device_index = config.get("playback_device")
        self.pa: Optional[object] = None

        if HAS_PYAUDIO:
            self.pa = pyaudio.PyAudio()
            self._find_device()

    def _find_device(self):
        """Auto-detect USB audio output device."""
        if self.device_index is not None:
            return

        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            max_out = info.get("maxOutputChannels", 0)
            if max_out > 0 and ("usb" in name or "pnp" in name):
                self.device_index = i
                logger.info(f"Auto-detected output device [{i}]: {info['name']}")
                return

        logger.warning("No USB audio output found, using system default")

    def play_file(self, path: str):
        """Play a WAV file through the speaker."""
        if not HAS_PYAUDIO:
            logger.warning(f"PyAudio not available, skipping playback of {path}")
            return

        wf = wave.open(path, "rb")
        stream = self.pa.open(
            format=self.pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=self.device_index,
        )

        chunk_size = 1024
        data = wf.readframes(chunk_size)
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)

        stream.stop_stream()
        stream.close()
        wf.close()
        logger.debug(f"Finished playing {path}")

    def play_raw(self, audio_data: bytes, sample_rate: int = 22050,
                 channels: int = 1, sample_width: int = 2):
        """Play raw audio bytes."""
        if not HAS_PYAUDIO:
            return

        stream = self.pa.open(
            format=self.pa.get_format_from_width(sample_width),
            channels=channels,
            rate=sample_rate,
            output=True,
            output_device_index=self.device_index,
        )
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()

    def close(self):
        if self.pa:
            self.pa.terminate()
            self.pa = None
