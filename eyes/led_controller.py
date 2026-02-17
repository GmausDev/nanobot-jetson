import logging
import time
from typing import Optional

logger = logging.getLogger("nanobot.eyes.led")

try:
    from rpi_ws281x import PixelStrip, Color
    HAS_LED = True
except ImportError:
    HAS_LED = False
    logger.warning("rpi_ws281x not available — LED control disabled")


class LEDController:
    """Low-level WS2812B LED strip controller."""

    def __init__(self, config: dict):
        self.num_leds = config.get("num_leds", 32)
        self.gpio_pin = config.get("gpio_pin", 18)
        self.brightness = int(config.get("brightness", 0.4) * 255)
        self.freq = config.get("led_freq", 800000)
        self.dma = config.get("dma_channel", 10)
        self.invert = config.get("invert", False)
        self.strip: Optional[object] = None

    def setup(self):
        """Initialize the LED strip."""
        if not HAS_LED:
            logger.info("LED hardware not available (dev mode)")
            return

        self.strip = PixelStrip(
            self.num_leds,
            self.gpio_pin,
            self.freq,
            self.dma,
            self.invert,
            self.brightness,
        )
        self.strip.begin()
        self.clear()
        logger.info(f"LED strip initialized: {self.num_leds} LEDs on GPIO{self.gpio_pin}")

    def set_pixel(self, index: int, r: int, g: int, b: int):
        """Set a single pixel color."""
        if self.strip:
            self.strip.setPixelColor(index, Color(r, g, b))

    def set_all(self, r: int, g: int, b: int):
        """Set all pixels to the same color."""
        for i in range(self.num_leds):
            self.set_pixel(i, r, g, b)
        self.show()

    def set_ring(self, ring: int, r: int, g: int, b: int):
        """Set all pixels on a specific ring (0 or 1)."""
        start = ring * 16
        for i in range(start, start + 16):
            self.set_pixel(i, r, g, b)

    def show(self):
        """Push pixel data to the strip."""
        if self.strip:
            self.strip.show()

    def clear(self):
        """Turn off all LEDs."""
        self.set_all(0, 0, 0)

    def set_brightness(self, brightness: float):
        """Set brightness (0.0-1.0)."""
        self.brightness = int(brightness * 255)
        if self.strip:
            self.strip.setBrightness(self.brightness)
            self.show()

    def shutdown(self):
        """Clean up — turn off LEDs."""
        self.clear()
        logger.info("LEDs turned off")
