# 🤖 NanoBot

A conversational desktop robot powered by **Jetson Nano 4GB**. It listens, thinks, and speaks — all running locally with no cloud APIs.

## What it does

**You talk → it listens → it thinks → it replies** with LED eyes showing what it's doing.

| Stage | Tool | Hardware |
|-------|------|----------|
| Hear | Whisper.cpp (STT) | USB Mic (Waveshare) |
| Think | llama.cpp + TinyLlama 1.1B | Jetson Nano GPU |
| Speak | Piper TTS | USB Speaker |
| Express | WS2812B NeoPixels | 2× 16-LED rings |

## Quick Start

### 1. Flash & Setup

```bash
# Flash JetPack 4.6.x to SD card, boot Jetson Nano
# SSH in, then:
git clone <this-repo> ~/nanobot
cd ~/nanobot
sudo bash scripts/setup.sh
```

This installs everything: whisper.cpp, llama.cpp, Piper, Python deps, downloads models, creates systemd services.

### 2. Test Hardware

```bash
# Test mic and speaker
python scripts/test_audio.py

# Test LED rings (needs sudo for GPIO)
sudo python scripts/test_leds.py

# Benchmark pipeline latency
python scripts/benchmark.py
```

### 3. Run

```bash
# Start LLM server (keep running in background)
sudo systemctl start llama-server

# Run NanoBot
sudo python main.py

# Or with debug logging:
sudo python main.py --debug
```

Press **Enter** to simulate wake word (OpenWakeWord replaces this with voice activation).

### 4. Auto-start on boot

```bash
sudo systemctl enable llama-server
sudo systemctl enable nanobot
```

## Project Structure

```
nanobot/
├── main.py              # Entry point
├── config.yaml          # All tunables
├── core/
│   ├── orchestrator.py  # State machine + async event loop
│   ├── states.py        # State enum + transitions
│   └── events.py        # Event types
├── audio/
│   ├── capture.py       # Mic recording (PyAudio)
│   ├── playback.py      # Speaker output
│   └── vad.py           # Voice Activity Detection
├── stt/
│   └── whisper_engine.py  # Whisper.cpp wrapper
├── llm/
│   ├── llama_client.py  # llama.cpp HTTP client
│   ├── prompt.py        # System prompt + history
│   └── streamer.py      # Sentence boundary detection
├── tts/
│   └── piper_engine.py  # Piper TTS wrapper
├── eyes/
│   ├── led_controller.py  # WS2812B driver
│   ├── animations.py     # All LED patterns
│   └── eye_manager.py    # State → animation mapping
└── scripts/
    ├── setup.sh         # One-shot installer
    ├── test_audio.py    # Audio hardware test
    ├── test_leds.py     # LED ring test
    └── benchmark.py     # Pipeline latency test
```

## Configuration

Edit `config.yaml` to tune:

- **Audio**: sample rate, device selection
- **VAD**: sensitivity, silence threshold
- **Wake Word**: model, threshold
- **STT**: Whisper model, language (en/es)
- **LLM**: model, GPU layers, temperature, system prompt
- **TTS**: voice model, speed
- **LEDs**: animations, colors, brightness

## LED States

| State | Animation | Color | Meaning |
|-------|-----------|-------|---------|
| 😴 IDLE | Breathing | Dim white | Waiting |
| ⭐ WAKE | Spiral | Gold | Wake word detected |
| 👂 LISTENING | Expanding | Cyan | Recording speech |
| 🧠 THINKING | Chase | Purple | Processing |
| 🗣️ SPEAKING | Pulse | Green | Playing response |
| ⚠️ ERROR | Flash | Red | Something broke |

## Switching Language

In `config.yaml`, change both:
```yaml
stt:
  language: "es"   # en or es
tts:
  language: "es"   # en or es
```

## Requirements

- Jetson Nano 4GB (B01) with JetPack 4.6.x
- 5V/4A barrel jack power supply (J48 jumper shorted)
- 64GB+ MicroSD card
- Waveshare USB Audio Codec (mic + speaker)
- 2× WS2812B 16-LED NeoPixel rings
- 3.3V→5V logic level shifter
- WiFi module (Intel AC8265)

## Memory Usage (~2.1GB / 4GB)

| Component | RAM |
|-----------|-----|
| OS (headless) | ~800MB |
| llama.cpp + TinyLlama Q4 | ~900MB |
| Whisper.cpp (tiny) | ~150MB |
| Piper TTS | ~100MB |
| Python + libs | ~150MB |

Run headless to stay under budget. 4GB swap file as safety net.

## License

MIT
