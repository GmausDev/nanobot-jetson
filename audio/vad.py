import logging
import time
from typing import Optional

logger = logging.getLogger("nanobot.audio.vad")

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    logger.warning("webrtcvad not available")


class VADProcessor:
    def __init__(self, vad_config: dict, audio_config: dict):
        self.aggressiveness = vad_config.get("aggressiveness", 2)
        self.silence_threshold = vad_config.get("silence_threshold", 1.5)
        self.min_speech = vad_config.get("min_speech_duration", 0.5)
        self.max_speech = vad_config.get("max_speech_duration", 30)
        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.chunk_size = audio_config.get("chunk_size", 480)  # 30ms at 16kHz

        self.vad = None
        if HAS_WEBRTCVAD:
            self.vad = webrtcvad.Vad(self.aggressiveness)

    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if an audio chunk contains speech."""
        if self.vad is None:
            return True  # fallback: assume everything is speech
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception:
            return False

    def record_until_silence(self, capture, output_path: str) -> bool:
        """
        Record audio until silence is detected after speech.
        Returns True if speech was captured, False if timeout/no speech.
        """
        from audio.capture import AudioCapture

        frames = []
        speech_started = False
        silence_start = None
        record_start = time.time()

        chunk_duration = self.chunk_size / self.sample_rate  # seconds per chunk

        capture.open_stream()

        try:
            while True:
                elapsed = time.time() - record_start

                # Max duration guard
                if elapsed > self.max_speech:
                    logger.warning("Max speech duration reached")
                    break

                chunk = capture.read_chunk()
                is_speech = self.is_speech(chunk)

                if is_speech:
                    speech_started = True
                    silence_start = None
                    frames.append(chunk)
                elif speech_started:
                    frames.append(chunk)  # keep recording during pauses
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) >= self.silence_threshold:
                        logger.info(f"Silence detected after {elapsed:.1f}s of recording")
                        break
                else:
                    # No speech yet — timeout after 10s of waiting
                    if elapsed > 10.0:
                        logger.warning("No speech detected, timing out")
                        return False

        finally:
            capture.close()

        # Check minimum speech duration
        total_speech = len(frames) * chunk_duration
        if total_speech < self.min_speech:
            logger.warning(f"Speech too short ({total_speech:.2f}s), ignoring")
            return False

        # Save to WAV
        AudioCapture.save_wav(
            frames, output_path,
            sample_rate=self.sample_rate,
            channels=1,
            sample_width=2
        )
        logger.info(f"Recorded {total_speech:.1f}s of speech → {output_path}")
        return True
