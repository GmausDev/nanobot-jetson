import logging
import threading
import time
from typing import Optional

from core.states import State
from eyes.led_controller import LEDController
from eyes.animations import Animation, create_animation

logger = logging.getLogger("nanobot.eyes.manager")


class EyeManager:
    """Maps robot states to LED animations and runs the render loop."""

    def __init__(self, config: dict):
        self.config = config
        self.controller = LEDController(config)
        self.animation_configs = config.get("animations", {})
        self.current_animation: Optional[Animation] = None
        self.current_state: Optional[State] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 60
        self._frame_time = 1.0 / self._fps

    def setup(self):
        """Initialize LED hardware and start render thread."""
        self.controller.setup()
        self.running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()
        logger.info("Eye manager started (render thread running)")

    def set_state(self, state: State):
        """Change the current animation based on robot state."""
        if state == self.current_state:
            return

        self.current_state = state
        state_name = state.name.lower()

        # Map state to animation config
        anim_cfg = self.animation_configs.get(state_name)
        if state == State.WAKE_DETECTED:
            anim_cfg = self.animation_configs.get("wakeup")

        if anim_cfg is None:
            logger.warning(f"No animation config for state: {state_name}")
            self.current_animation = None
            self.controller.clear()
            return

        pattern = anim_cfg.get("pattern", "solid")
        color = tuple(anim_cfg.get("color", [255, 255, 255]))
        speed = anim_cfg.get("speed", 0.05)

        self.current_animation = create_animation(
            pattern=pattern,
            num_leds=self.config.get("num_leds", 32),
            color=color,
            speed=speed,
        )
        self.current_animation.reset()
        logger.debug(f"Animation → {pattern} ({color}) for state {state_name}")

    def set_amplitude(self, amplitude: float):
        """Set audio amplitude for speaking animation."""
        if self.current_animation and hasattr(self.current_animation, 'set_amplitude'):
            self.current_animation.set_amplitude(amplitude)

    def _render_loop(self):
        """Background thread: render animation frames at target FPS."""
        while self.running:
            start = time.time()

            if self.current_animation:
                pixels = self.current_animation.tick()
                for i, (r, g, b) in enumerate(pixels):
                    self.controller.set_pixel(i, r, g, b)
                self.controller.show()

            # Frame timing
            elapsed = time.time() - start
            sleep_time = self._frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def shutdown(self):
        """Stop render thread and turn off LEDs."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.controller.shutdown()
        logger.info("Eye manager stopped")
