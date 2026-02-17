#!/usr/bin/env python3
"""
NanoBot — Conversational Robot on Jetson Nano
Entry point: starts the orchestrator event loop.

Usage:
    python main.py                  # Normal mode (wake word via Enter key)
    python main.py --config custom.yaml  # Custom config
    python main.py --debug          # Debug logging
"""

import asyncio
import argparse
import logging
import sys
import signal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import Orchestrator


def setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s [%(levelname)5s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=getattr(logging, level), format=fmt, datefmt=datefmt)
    # Quiet down noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="NanoBot — Conversational Robot")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = "DEBUG" if args.debug else "INFO"
    setup_logging(level)

    logger = logging.getLogger("nanobot")
    logger.info("=" * 50)
    logger.info("  🤖 NanoBot starting up...")
    logger.info("=" * 50)

    # Check config exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    bot = Orchestrator(config_path=args.config)

    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.run_until_complete(bot.cleanup())
        loop.close()
        logger.info("NanoBot shut down. Goodbye! 🤖")


if __name__ == "__main__":
    main()
