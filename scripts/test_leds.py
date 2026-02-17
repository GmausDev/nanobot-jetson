#!/usr/bin/env python3
"""
Test LED rings: cycle through all animation patterns.
Run: sudo python scripts/test_leds.py

NOTE: Must run as root (sudo) for GPIO access on Jetson.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eyes.led_controller import LEDController
from eyes.animations import create_animation, ANIMATION_CLASSES

# Config matching our hardware
LED_CONFIG = {
    "num_leds": 32,
    "gpio_pin": 18,
    "brightness": 0.3,
    "led_freq": 800000,
    "dma_channel": 10,
    "invert": False,
}

DEMO_ANIMATIONS = [
    ("breathe",  (40, 40, 50),    0.02, "IDLE — Breathing (dim white)"),
    ("expand",   (0, 200, 220),   0.05, "LISTENING — Expanding (cyan)"),
    ("chase",    (140, 80, 255),  0.08, "THINKING — Chase (purple)"),
    ("pulse",    (0, 230, 100),   0.06, "SPEAKING — Pulse (green)"),
    ("flash",    (255, 50, 50),   0.15, "ERROR — Flash (red)"),
    ("spiral",   (255, 200, 0),   0.10, "WAKE UP — Spiral (gold)"),
    ("solid",    (0, 100, 255),   0.00, "Solid blue test"),
]


def main():
    print("=" * 50)
    print("  🤖 NanoBot LED Test")
    print("=" * 50)

    controller = LEDController(LED_CONFIG)

    try:
        controller.setup()
    except Exception as e:
        print(f"\n⚠️  LED hardware not available: {e}")
        print("Running in simulation mode (no physical LEDs)")
        print("Showing animation frames as text...\n")
        run_simulation()
        return

    print(f"\nInitialized {LED_CONFIG['num_leds']} LEDs on GPIO{LED_CONFIG['gpio_pin']}")
    print("Running animation demos (5s each)...\n")

    for pattern, color, speed, desc in DEMO_ANIMATIONS:
        print(f"  ▶ {desc}")
        anim = create_animation(pattern, num_leds=32, color=color, speed=speed)

        start = time.time()
        while time.time() - start < 5.0:
            pixels = anim.tick()
            for i, (r, g, b) in enumerate(pixels):
                controller.set_pixel(i, r, g, b)
            controller.show()
            time.sleep(1 / 60)  # 60 FPS

    # Cleanup
    controller.shutdown()
    print("\n✅ LED test complete! All animations working.")


def run_simulation():
    """Text-based simulation when no hardware is available."""
    for pattern, color, speed, desc in DEMO_ANIMATIONS:
        print(f"  ▶ {desc}")
        anim = create_animation(pattern, num_leds=32, color=color, speed=speed)

        # Show 3 frames
        for frame in range(3):
            pixels = anim.tick()
            # Show as ring visualization
            ring1 = pixels[:16]
            ring2 = pixels[16:]
            r1_str = " ".join(f"{'●' if sum(p)>30 else '○'}" for p in ring1)
            r2_str = " ".join(f"{'●' if sum(p)>30 else '○'}" for p in ring2)
            print(f"    Ring L: {r1_str}")
            print(f"    Ring R: {r2_str}")
            time.sleep(0.1)
        print()

    print("✅ Simulation complete!")


if __name__ == "__main__":
    main()
