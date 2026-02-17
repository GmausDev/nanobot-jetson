#!/usr/bin/env python3
"""
Benchmark each pipeline stage to measure latency.
Run: python scripts/benchmark.py

Tests:
  1. Whisper STT transcription speed
  2. LLM inference speed (tokens/sec)
  3. Piper TTS synthesis speed
  4. End-to-end pipeline latency estimate
"""

import sys
import os
import time
import json
import subprocess
import shutil
import tempfile
import wave
import struct
import math
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_test_audio(path: str, duration: float = 3.0, sample_rate: int = 16000):
    """Generate a simple test WAV file with a tone."""
    num_samples = int(sample_rate * duration)
    freq = 440  # Hz
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            sample = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))
    return path


def bench_whisper():
    """Benchmark Whisper STT."""
    print("\n📊 Benchmark: Whisper STT")
    print("-" * 40)

    whisper_bin = shutil.which("whisper-cli")
    if not whisper_bin:
        print("  ⚠️  whisper-cli not found, skipping")
        return None

    model = "models/whisper/ggml-tiny.en.bin"
    if not os.path.exists(model):
        print(f"  ⚠️  Model not found: {model}")
        return None

    # Generate test audio
    test_audio = "/tmp/nanobot/bench_audio.wav"
    os.makedirs("/tmp/nanobot", exist_ok=True)
    generate_test_audio(test_audio, duration=3.0)

    # Warm-up run
    subprocess.run(
        [whisper_bin, "-m", model, "-f", test_audio, "--no-timestamps"],
        capture_output=True, timeout=60,
    )

    # Timed runs
    times = []
    for i in range(3):
        start = time.time()
        result = subprocess.run(
            [whisper_bin, "-m", model, "-f", test_audio, "--no-timestamps"],
            capture_output=True, text=True, timeout=60,
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.2f}s for 3s audio (ratio: {avg/3:.2f}x realtime)")
    return avg


def bench_llm():
    """Benchmark LLM inference speed."""
    print("\n📊 Benchmark: LLM (llama.cpp)")
    print("-" * 40)

    server_url = "http://127.0.0.1:8080"

    # Check if server is running
    try:
        urllib.request.urlopen(f"{server_url}/health", timeout=3)
    except Exception:
        print("  ⚠️  llama-server not running!")
        print("  Start with: llama-server -m models/llm/tinyllama-*.gguf --port 8080 -ngl 24")
        return None

    test_prompts = [
        "Hello, how are you?",
        "What is the weather like today?",
        "Tell me a short joke.",
    ]

    total_tokens = 0
    total_time = 0

    for prompt in test_prompts:
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": "You are a helpful robot. Be concise."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{server_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        start = time.time()
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.time() - start

        content = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("completion_tokens", len(content.split()))
        tps = tokens / elapsed if elapsed > 0 else 0

        total_tokens += tokens
        total_time += elapsed

        print(f"  \"{prompt[:30]}...\" → {tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"  Average: {avg_tps:.1f} tokens/sec")
    return avg_tps


def bench_piper():
    """Benchmark Piper TTS."""
    print("\n📊 Benchmark: Piper TTS")
    print("-" * 40)

    piper_bin = shutil.which("piper")
    if not piper_bin:
        print("  ⚠️  piper not found, skipping")
        return None

    model = "models/piper/en_US-lessac-medium.onnx"
    if not os.path.exists(model):
        print(f"  ⚠️  Model not found: {model}")
        return None

    test_texts = [
        "Hello there.",
        "I am NanoBot, a small desktop robot with LED eyes.",
        "The weather today is quite nice, don't you think? I wouldn't know, I don't have sensors for that.",
    ]

    times = []
    for text in test_texts:
        out = f"/tmp/nanobot/bench_tts_{len(times)}.wav"
        start = time.time()
        result = subprocess.run(
            [piper_bin, "--model", model, "--output_file", out],
            input=text, capture_output=True, text=True, timeout=30,
        )
        elapsed = time.time() - start

        # Get audio duration
        if os.path.exists(out):
            with wave.open(out) as wf:
                audio_dur = wf.getnframes() / wf.getframerate()
            rtf = elapsed / audio_dur if audio_dur > 0 else 0
            print(f"  \"{text[:40]}...\" → {elapsed:.2f}s synth, {audio_dur:.1f}s audio (RTF: {rtf:.2f})")
        else:
            print(f"  \"{text[:40]}...\" → {elapsed:.2f}s (output missing)")

        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"  Average synthesis time: {avg:.2f}s")
    return avg


def main():
    print("=" * 50)
    print("  🤖 NanoBot Pipeline Benchmark")
    print("=" * 50)

    stt_time = bench_whisper()
    llm_tps = bench_llm()
    tts_time = bench_piper()

    # Estimate end-to-end
    print("\n" + "=" * 50)
    print("  📋 End-to-End Latency Estimate")
    print("=" * 50)

    stt_est = stt_time if stt_time else 1.0
    llm_est = (50 / llm_tps) if llm_tps else 5.0  # ~50 tokens for first sentence
    tts_est = tts_time if tts_time else 0.3

    total = stt_est + llm_est + tts_est
    print(f"  STT:      ~{stt_est:.1f}s")
    print(f"  LLM (1st): ~{llm_est:.1f}s (50 tokens)")
    print(f"  TTS:      ~{tts_est:.1f}s")
    print(f"  ─────────────────")
    print(f"  TOTAL:    ~{total:.1f}s from end of speech to robot speaking")
    print()

    if total < 3:
        print("  🟢 Excellent! Under 3s response time")
    elif total < 5:
        print("  🟡 Good. Under 5s — acceptable for conversation")
    else:
        print("  🔴 Slow. Consider smaller model or optimization")


if __name__ == "__main__":
    main()
