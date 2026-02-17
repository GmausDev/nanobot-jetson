import math
import time
import logging
from typing import Optional

logger = logging.getLogger("nanobot.eyes.animations")

try:
    from rpi_ws281x import Color
    HAS_LED = True
except ImportError:
    HAS_LED = False
    def Color(r, g, b):
        return (r << 16) | (g << 8) | b


class Animation:
    """Base class for LED animations."""

    def __init__(self, num_leds: int = 32, color: tuple = (255, 255, 255), speed: float = 0.05):
        self.num_leds = num_leds
        self.color = color  # (r, g, b)
        self.speed = speed
        self.frame = 0
        self.start_time = time.time()

    def tick(self) -> list[tuple[int, int, int]]:
        """
        Compute one frame. Returns list of (r, g, b) tuples, one per LED.
        Override in subclasses.
        """
        self.frame += 1
        return [(0, 0, 0)] * self.num_leds

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def reset(self):
        self.frame = 0
        self.start_time = time.time()


class BreatheAnimation(Animation):
    """Slow breathing pulse — all LEDs fade in and out. Used for IDLE state."""

    def tick(self) -> list[tuple[int, int, int]]:
        self.frame += 1
        t = self.elapsed() * self.speed * 10
        # Sine wave for smooth breathing (0.15 to 1.0 range)
        brightness = 0.15 + 0.85 * ((math.sin(t) + 1) / 2)
        r = int(self.color[0] * brightness)
        g = int(self.color[1] * brightness)
        b = int(self.color[2] * brightness)
        return [(r, g, b)] * self.num_leds


class ExpandAnimation(Animation):
    """Expanding ring effect — LEDs light up sequentially outward. Used for LISTENING."""

    def tick(self) -> list[tuple[int, int, int]]:
        self.frame += 1
        t = self.elapsed() * self.speed * 20
        pixels = [(0, 0, 0)] * self.num_leds

        # Expanding wave on each ring (16 LEDs per ring)
        for ring in range(2):
            offset = ring * 16
            active = int(t % 16)
            for i in range(16):
                dist = min(abs(i - active), 16 - abs(i - active))
                fade = max(0, 1.0 - dist / 4.0)
                r = int(self.color[0] * fade)
                g = int(self.color[1] * fade)
                b = int(self.color[2] * fade)
                pixels[offset + i] = (r, g, b)

        return pixels


class ChaseAnimation(Animation):
    """Spinning chase light — dot rotates around ring. Used for THINKING."""

    def tick(self) -> list[tuple[int, int, int]]:
        self.frame += 1
        t = self.elapsed() * self.speed * 30
        pixels = [(0, 0, 0)] * self.num_leds

        for ring in range(2):
            offset = ring * 16
            pos = int(t) % 16
            # Trail of 4 LEDs with decay
            for trail in range(5):
                idx = (pos - trail) % 16
                fade = 1.0 - (trail / 5.0)
                r = int(self.color[0] * fade)
                g = int(self.color[1] * fade)
                b = int(self.color[2] * fade)
                pixels[offset + idx] = (r, g, b)

        return pixels


class PulseAnimation(Animation):
    """
    Quick pulsing wave — brightness modulated by audio amplitude.
    Used for SPEAKING. Amplitude can be set externally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amplitude = 0.5  # 0.0-1.0, set externally from audio

    def set_amplitude(self, amplitude: float):
        self.amplitude = max(0.0, min(1.0, amplitude))

    def tick(self) -> list[tuple[int, int, int]]:
        self.frame += 1
        t = self.elapsed() * self.speed * 15
        pixels = [(0, 0, 0)] * self.num_leds

        base = 0.3 + 0.7 * self.amplitude

        for ring in range(2):
            offset = ring * 16
            for i in range(16):
                wave = (math.sin(t + i * 0.5) + 1) / 2
                brightness = base * (0.5 + 0.5 * wave)
                r = int(self.color[0] * brightness)
                g = int(self.color[1] * brightness)
                b = int(self.color[2] * brightness)
                pixels[offset + i] = (r, g, b)

        return pixels


class FlashAnimation(Animation):
    """Quick red flash — used for ERROR state."""

    def tick(self) -> list[tuple[int, int, int]]:
        self.frame += 1
        t = self.elapsed()
        # Flash on/off every 0.2s
        on = (int(t / 0.2) % 2) == 0
        if on:
            return [self.color] * self.num_leds
        else:
            return [(0, 0, 0)] * self.num_leds


class SpiralAnimation(Animation):
    """Spiral wind-up effect — used for WAKE_DETECTED acknowledgment."""

    def tick(self) -> list[tuple[int, int, int]]:
        self.frame += 1
        t = self.elapsed() * self.speed * 40
        pixels = [(0, 0, 0)] * self.num_leds

        for ring in range(2):
            offset = ring * 16
            filled = min(int(t) % 17, 16)
            for i in range(filled):
                progress = i / 16.0
                r = int(self.color[0] * progress)
                g = int(self.color[1] * progress)
                b = int(self.color[2] * progress)
                pixels[offset + i] = (r, g, b)

        return pixels


class SolidAnimation(Animation):
    """Static solid color — utility animation."""

    def tick(self) -> list[tuple[int, int, int]]:
        return [self.color] * self.num_leds


# --- Animation Registry ---
ANIMATION_CLASSES = {
    "breathe": BreatheAnimation,
    "expand": ExpandAnimation,
    "chase": ChaseAnimation,
    "pulse": PulseAnimation,
    "flash": FlashAnimation,
    "spiral": SpiralAnimation,
    "solid": SolidAnimation,
}


def create_animation(pattern: str, num_leds: int = 32,
                     color: tuple = (255, 255, 255),
                     speed: float = 0.05) -> Animation:
    """Factory function to create an animation by name."""
    cls = ANIMATION_CLASSES.get(pattern, SolidAnimation)
    return cls(num_leds=num_leds, color=tuple(color), speed=speed)
