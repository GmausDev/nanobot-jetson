from __future__ import annotations

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Optional

from core.states import State, get_next_state
from core.events import Event, error_occurred, recover, ready, audio_finished
from audio.capture import AudioCapture
from audio.playback import AudioPlayback
from audio.vad import VADProcessor
from stt.whisper_engine import WhisperSTT
from llm.llama_client import LlamaClient
from llm.streamer import SentenceStreamer
from llm.prompt import PromptManager
from tts.piper_engine import PiperTTS
from eyes.eye_manager import EyeManager
from audio.wake_word import WakeWordDetector

logger = logging.getLogger("nanobot.orchestrator")


class Orchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.state = State.IDLE
        self.event_queue: asyncio.Queue[Event] | None = None  # Created in setup()
        self.running = False

        # Components (initialized in setup())
        self.capture: Optional[AudioCapture] = None
        self.playback: Optional[AudioPlayback] = None
        self.vad: Optional[VADProcessor] = None
        self.stt: Optional[WhisperSTT] = None
        self.llm: Optional[LlamaClient] = None
        self.tts: Optional[PiperTTS] = None
        self.eyes: Optional[EyeManager] = None
        self.prompt_mgr: Optional[PromptManager] = None
        self.streamer: Optional[SentenceStreamer] = None
        self.wake_detector: Optional[WakeWordDetector] = None

    async def setup(self):
        """Initialize all components."""
        logger.info("Initializing NanoBot components...")

        # Create event queue in the running event loop (Python 3.8 compat)
        self.event_queue = asyncio.Queue()

        cfg = self.config

        # Audio
        self.capture = AudioCapture(cfg["audio"])
        self.playback = AudioPlayback(cfg["audio"])

        # VAD
        self.vad = VADProcessor(cfg["vad"], cfg["audio"])

        # STT
        self.stt = WhisperSTT(cfg["stt"])

        # LLM
        self.llm = LlamaClient(cfg["llm"])
        self.prompt_mgr = PromptManager(cfg["llm"])
        self.streamer = SentenceStreamer()

        # TTS
        self.tts = PiperTTS(cfg["tts"])

        # Eyes
        if cfg["eyes"]["enabled"]:
            self.eyes = EyeManager(cfg["eyes"])
            self.eyes.setup()

        # Wake word detector (uses its own AudioCapture + VAD)
        if cfg["wake_word"].get("enabled", True):
            self.wake_detector = WakeWordDetector(
                wake_config=cfg["wake_word"],
                audio_config=cfg["audio"],
                vad_config=cfg["vad"],
                stt=self.stt,
            )

        # Create tmp dir
        Path(cfg["system"]["tmp_dir"]).mkdir(parents=True, exist_ok=True)

        logger.info("All components initialized ✓")

    async def run(self):
        """Main event loop."""
        await self.setup()
        self.running = True
        logger.info("NanoBot is alive! Entering main loop...")

        # Set initial LED state
        await self._set_state(State.IDLE)

        # Start background tasks
        tasks = [
            asyncio.create_task(self._wake_word_loop()),
            asyncio.create_task(self._event_processor()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Shutting down...")
        finally:
            self.running = False
            await self.cleanup()

    async def _event_processor(self):
        """Process events from the queue and drive state transitions."""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            logger.debug(f"Event: {event.name} (state={self.state.name})")

            next_state = get_next_state(self.state, event.name)
            if next_state is None:
                logger.warning(f"Invalid transition: {self.state.name} + {event.name}")
                continue

            await self._set_state(next_state)
            await self._handle_state(next_state, event)

    async def _set_state(self, new_state: State):
        """Update state and LED animation."""
        old = self.state
        self.state = new_state
        logger.info(f"State: {old.name} → {new_state.name}")

        if self.eyes:
            self.eyes.set_state(new_state)

    async def _handle_state(self, state: State, event: Event):
        """Execute actions for each state."""
        try:
            if state == State.WAKE_DETECTED:
                await self._on_wake_detected()
            elif state == State.LISTENING:
                await self._on_listening()
            elif state == State.THINKING:
                await self._on_thinking(event)
            elif state == State.SPEAKING:
                await self._on_speaking(event)
            elif state == State.ERROR:
                await self._on_error(event)
            elif state == State.IDLE:
                pass  # just chill
        except Exception as e:
            logger.error(f"Error in state {state.name}: {e}", exc_info=True)
            await self.event_queue.put(error_occurred(str(e), e))

    # --- Wake Word Loop ---
    async def _wake_word_loop(self):
        """Continuously listen for wake word when IDLE."""
        if self.wake_detector is None:
            logger.warning("Wake word detector not initialized, waiting for Enter key")
            await self._fallback_wake_loop()
            return

        logger.info("Wake word listener started (whisper-based)")

        while self.running:
            if self.state != State.IDLE:
                await asyncio.sleep(0.1)
                continue

            if not self.wake_detector.check_cooldown():
                await asyncio.sleep(0.1)
                continue

            try:
                detected, confidence, transcript = await asyncio.get_event_loop().run_in_executor(
                    None, self.wake_detector.listen_once
                )
                if detected and self.state == State.IDLE:
                    self.wake_detector.mark_detected()
                    from core.events import wake_word_detected
                    await self.event_queue.put(wake_word_detected(confidence))
            except Exception as e:
                logger.error(f"Wake word detection error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _fallback_wake_loop(self):
        """Fallback: use Enter key as wake trigger (for development)."""
        logger.info("Fallback wake word: press Enter to trigger")
        while self.running:
            if self.state != State.IDLE:
                await asyncio.sleep(0.1)
                continue
            try:
                await asyncio.get_event_loop().run_in_executor(None, input)
                if self.state == State.IDLE:
                    from core.events import wake_word_detected
                    await self.event_queue.put(wake_word_detected(1.0))
            except Exception:
                await asyncio.sleep(0.1)

    # --- State Handlers ---

    async def _on_wake_detected(self):
        """Play a small acknowledgment and start listening."""
        logger.info("Wake word detected! Starting to listen...")
        # Brief LED animation, then transition to LISTENING
        await asyncio.sleep(0.3)
        await self.event_queue.put(ready())

    async def _on_listening(self):
        """Record audio until silence detected."""
        logger.info("Listening for speech...")
        tmp_dir = self.config["system"]["tmp_dir"]
        audio_path = f"{tmp_dir}/recording.wav"

        # Record with VAD
        recorded = await asyncio.get_event_loop().run_in_executor(
            None, self._record_speech, audio_path
        )

        if recorded:
            from core.events import silence_detected
            await self.event_queue.put(silence_detected())
        else:
            from core.events import timeout
            await self.event_queue.put(timeout())

    def _record_speech(self, output_path: str) -> bool:
        """Blocking: Record speech using VAD. Returns True if speech was captured."""
        return self.vad.record_until_silence(
            capture=self.capture,
            output_path=output_path,
        )

    async def _on_thinking(self, event: Event):
        """Transcribe audio, send to LLM, stream response."""
        tmp_dir = self.config["system"]["tmp_dir"]
        audio_path = f"{tmp_dir}/recording.wav"

        # Step 1: STT
        logger.info("Transcribing speech...")
        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self.stt.transcribe, audio_path
        )

        if not transcript or not transcript.strip():
            logger.warning("Empty transcript, returning to idle")
            from core.events import error_occurred
            await self.event_queue.put(error_occurred("Empty transcript"))
            return

        logger.info(f"User said: \"{transcript}\"")

        # Step 2: Build prompt with history
        messages = self.prompt_mgr.build_messages(transcript)

        # Step 3: Stream LLM response, collect sentences
        logger.info("Generating response...")
        full_response = ""
        self.streamer.reset()

        async for token in self.llm.stream_completion(messages):
            full_response += token
            sentences = self.streamer.feed(token)
            for sentence in sentences:
                logger.info(f"Sentence ready: \"{sentence}\"")

        # Flush remaining text
        remaining = self.streamer.flush()
        if remaining:
            full_response = full_response  # already accumulated

        logger.info(f"Bot response: \"{full_response}\"")

        # Update conversation history
        self.prompt_mgr.add_turn(transcript, full_response)

        # Transition to speaking
        from core.events import response_ready
        await self.event_queue.put(response_ready(full_response))

    async def _on_speaking(self, event: Event):
        """Synthesize and play the response."""
        text = event.data["text"]
        tmp_dir = self.config["system"]["tmp_dir"]
        audio_path = f"{tmp_dir}/response.wav"

        # TTS
        logger.info("Synthesizing speech...")
        await asyncio.get_event_loop().run_in_executor(
            None, self.tts.synthesize, text, audio_path
        )

        # Play audio
        logger.info("Playing response...")
        await asyncio.get_event_loop().run_in_executor(
            None, self.playback.play_file, audio_path
        )

        await self.event_queue.put(audio_finished())

    async def _on_error(self, event: Event):
        """Handle error state — flash LEDs red, then recover."""
        msg = event.data.get("message", "Unknown error") if event.data else "Unknown"
        logger.error(f"Error state: {msg}")
        await asyncio.sleep(2.0)
        await self.event_queue.put(recover())

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        if self.capture:
            self.capture.close()
        if self.playback:
            self.playback.close()
        if self.wake_detector:
            self.wake_detector.close()
        if self.eyes:
            self.eyes.shutdown()
        logger.info("Goodbye! 🤖")
