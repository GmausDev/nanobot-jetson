from __future__ import annotations

import logging
import time
from pathlib import Path

from audio.capture import AudioCapture
from audio.vad import VADProcessor

logger = logging.getLogger("nanobot.audio.wake_word")


class WakeWordDetector:
    """Detect wake words by recording short VAD bursts and transcribing with whisper."""

    def __init__(self, wake_config: dict, audio_config: dict, vad_config: dict, stt):
        self.phrases = [p.lower() for p in wake_config.get("phrases", ["hey jarvis"])]
        self.cooldown = wake_config.get("cooldown", 2.0)
        self.listen_duration = wake_config.get("listen_duration", 3.0)
        self.tmp_dir = "/tmp/nanobot"

        # Own AudioCapture and VAD (separate from conversation pipeline)
        self.capture = AudioCapture(audio_config)
        self.vad = VADProcessor(vad_config, audio_config)
        self.stt = stt

        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.chunk_size = audio_config.get("chunk_size", 480)
        self._last_detection = 0.0

        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

    def listen_once(self) -> tuple:
        """
        Blocking: wait for speech via VAD, record a short burst, transcribe,
        and check for wake phrase.

        Returns (detected: bool, confidence: float, transcript: str).
        """
        audio_path = f"{self.tmp_dir}/wake_word.wav"
        frames = []
        speech_started = False
        silence_start = None
        record_start = None

        chunk_duration = self.chunk_size / self.sample_rate
        # Shorter silence threshold for wake word — just need a brief phrase
        silence_threshold = 0.8

        self.capture.open_stream()

        try:
            wait_start = time.time()
            while True:
                chunk = self.capture.read_chunk()
                is_speech = self.vad.is_speech(chunk)

                if is_speech:
                    if not speech_started:
                        speech_started = True
                        record_start = time.time()
                        logger.debug("Wake word VAD: speech started")
                    silence_start = None
                    frames.append(chunk)
                elif speech_started:
                    frames.append(chunk)
                    if silence_start is None:
                        silence_start = time.time()
                    elif (time.time() - silence_start) >= silence_threshold:
                        logger.debug("Wake word VAD: silence after speech")
                        break

                # Cap recording length
                if record_start and (time.time() - record_start) >= self.listen_duration:
                    logger.debug("Wake word VAD: max listen duration reached")
                    break

                # Don't wait forever for speech to start (15s timeout)
                if not speech_started and (time.time() - wait_start) > 15.0:
                    return (False, 0.0, "")

        finally:
            self.capture.close_stream()

        # Need at least ~0.3s of audio to be worth transcribing
        total_duration = len(frames) * chunk_duration
        if total_duration < 0.3:
            return (False, 0.0, "")

        # Save and transcribe
        AudioCapture.save_wav(
            frames, audio_path,
            sample_rate=self.sample_rate,
            channels=1,
            sample_width=2,
        )

        transcript = self.stt.transcribe(audio_path)
        if not transcript:
            return (False, 0.0, "")

        transcript_lower = transcript.lower().strip()
        logger.debug(f"Wake word transcript: \"{transcript}\"")

        # Check against known phrases
        detected, confidence = self._match_phrases(transcript_lower)
        if detected:
            logger.info(f"Wake word detected! phrase=\"{transcript}\" confidence={confidence:.2f}")
        return (detected, confidence, transcript)

    def _match_phrases(self, transcript: str) -> tuple:
        """Check if transcript contains any wake phrase. Returns (matched, confidence)."""
        for phrase in self.phrases:
            if phrase in transcript:
                return (True, 1.0)

        # Fuzzy: check if most words of any phrase appear in the transcript
        transcript_words = set(transcript.split())
        for phrase in self.phrases:
            phrase_words = phrase.split()
            if len(phrase_words) == 0:
                continue
            matches = sum(1 for w in phrase_words if w in transcript_words)
            ratio = matches / len(phrase_words)
            if ratio >= 0.5 and len(phrase_words) > 1:
                return (True, ratio)

        return (False, 0.0)

    def check_cooldown(self) -> bool:
        """Returns True if enough time has passed since last detection."""
        return (time.time() - self._last_detection) >= self.cooldown

    def mark_detected(self):
        """Record detection timestamp for cooldown."""
        self._last_detection = time.time()

    def close(self):
        """Clean up resources."""
        self.capture.close()
