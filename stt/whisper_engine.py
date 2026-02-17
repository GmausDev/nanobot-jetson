import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger("nanobot.stt.whisper")


class WhisperSTT:
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "models/whisper/ggml-tiny.en.bin")
        self.model_path_es = config.get("model_path_es", "models/whisper/ggml-tiny.bin")
        self.language = config.get("language", "en")
        self.beam_size = config.get("beam_size", 2)
        self.best_of = config.get("best_of", 2)
        self.threads = config.get("threads", 4)
        self.use_gpu = config.get("use_gpu", True)

        # Find whisper binary
        self.binary = self._find_binary()

    def _find_binary(self) -> str:
        """Locate the whisper.cpp binary."""
        candidates = [
            "whisper-cli",                          # if installed globally
            "/usr/local/bin/whisper-cli",
            "build/bin/whisper-cli",                # local build
            str(Path.home() / "whisper.cpp/build/bin/whisper-cli"),
        ]
        for c in candidates:
            if shutil.which(c) or Path(c).exists():
                logger.info(f"Found whisper binary: {c}")
                return c

        logger.warning("whisper-cli not found — STT will fail until installed")
        return "whisper-cli"

    def _get_model_path(self) -> str:
        """Get model path based on current language."""
        if self.language == "es":
            return self.model_path_es
        return self.model_path

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a WAV file to text using whisper.cpp CLI."""
        model = self._get_model_path()

        cmd = [
            self.binary,
            "-m", model,
            "-f", audio_path,
            "-l", self.language if self.language != "auto" else "auto",
            "-t", str(self.threads),
            "--beam-size", str(self.beam_size),
            "--best-of", str(self.best_of),
            "--no-timestamps",
            "--print-special", "false",
        ]

        if self.use_gpu:
            cmd.extend(["--gpu"])

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Whisper error: {result.stderr}")
                return ""

            # Parse output — whisper-cli outputs text to stdout
            text = result.stdout.strip()

            # Clean up common whisper artifacts
            text = self._clean_transcript(text)

            return text

        except subprocess.TimeoutExpired:
            logger.error("Whisper transcription timed out")
            return ""
        except FileNotFoundError:
            logger.error(f"Whisper binary not found: {self.binary}")
            return ""

    def _clean_transcript(self, text: str) -> str:
        """Remove common whisper artifacts from transcript."""
        # Remove [BLANK_AUDIO] and similar tags
        import re
        text = re.sub(r'\[.*?\]', '', text)
        # Remove leading/trailing whitespace and extra spaces
        text = ' '.join(text.split())
        # Remove common hallucinations on silence
        noise_phrases = [
            "thank you", "thanks for watching", "subscribe",
            "you", "bye", "the end",
        ]
        if text.lower().strip() in noise_phrases:
            return ""
        return text.strip()

    def set_language(self, language: str):
        """Switch language (en/es/auto)."""
        self.language = language
        logger.info(f"STT language set to: {language}")
