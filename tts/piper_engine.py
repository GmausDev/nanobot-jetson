from __future__ import annotations

import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger("nanobot.tts.piper")


class PiperTTS:
    def __init__(self, config: dict):
        self.model_path = config.get("model_path", "models/piper/en_US-lessac-medium.onnx")
        self.model_config = config.get("model_config", "")
        self.model_path_es = config.get("model_path_es", "")
        self.model_config_es = config.get("model_config_es", "")
        self.language = config.get("language", "en")
        self.length_scale = config.get("length_scale", 1.0)
        self.noise_scale = config.get("noise_scale", 0.667)
        self.noise_w = config.get("noise_w", 0.8)
        self.speaker_id = config.get("speaker_id")

        self.binary = self._find_binary()

    def _find_binary(self) -> str:
        """Locate the piper binary."""
        candidates = [
            "piper",
            "/usr/local/bin/piper",
            str(Path.home() / "piper/piper"),
            "build/piper",
        ]
        for c in candidates:
            if shutil.which(c) or Path(c).exists():
                logger.info(f"Found piper binary: {c}")
                return c

        logger.warning("piper not found — TTS will fail until installed")
        return "piper"

    def _get_model_path(self) -> str:
        if self.language == "es" and self.model_path_es:
            return self.model_path_es
        return self.model_path

    def _get_config_path(self) -> str:
        if self.language == "es" and self.model_config_es:
            return self.model_config_es
        return self.model_config

    def synthesize(self, text: str, output_path: str) -> bool:
        """Synthesize text to a WAV file using Piper CLI."""
        if not text.strip():
            logger.warning("Empty text, skipping synthesis")
            return False

        model = self._get_model_path()

        cmd = [
            self.binary,
            "--model", model,
            "--output_file", output_path,
            "--length_scale", str(self.length_scale),
            "--noise_scale", str(self.noise_scale),
            "--noise_w", str(self.noise_w),
        ]

        config = self._get_config_path()
        if config:
            cmd.extend(["--config", config])

        if self.speaker_id is not None:
            cmd.extend(["--speaker", str(self.speaker_id)])

        logger.debug(f"Running: echo '{text[:50]}...' | {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Piper error: {result.stderr}")
                return False

            logger.debug(f"Synthesized to {output_path}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out")
            return False
        except FileNotFoundError:
            logger.error(f"Piper binary not found: {self.binary}")
            return False

    def synthesize_to_raw(self, text: str) -> bytes | None:
        """Synthesize text and return raw audio bytes."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            if self.synthesize(text, tmp.name):
                import wave
                with wave.open(tmp.name, "rb") as wf:
                    return wf.readframes(wf.getnframes())
        return None

    def set_language(self, language: str):
        """Switch language (en/es)."""
        self.language = language
        logger.info(f"TTS language set to: {language}")
