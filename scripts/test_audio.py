#!/usr/bin/env python3
"""
Test audio hardware: record from mic, play back through speaker.
Run: python scripts/test_audio.py
"""

import sys
import os
import time
import wave
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TMP_FILE = "/tmp/nanobot/test_recording.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
CHUNK = 1024


def list_devices():
    """List all available audio devices."""
    import pyaudio
    pa = pyaudio.PyAudio()
    print("\n🔊 Audio Devices:")
    print("-" * 60)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        direction = ""
        if info["maxInputChannels"] > 0:
            direction += "🎤 IN"
        if info["maxOutputChannels"] > 0:
            direction += " 🔈 OUT"
        print(f"  [{i}] {info['name']:40s} {direction}")
    print("-" * 60)
    pa.terminate()


def test_record():
    """Record audio from microphone."""
    import pyaudio

    os.makedirs(os.path.dirname(TMP_FILE), exist_ok=True)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"\n🎙️  Recording {RECORD_SECONDS}s... Speak now!")
    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        # Simple level meter
        samples = struct.unpack(f"<{CHUNK}h", data)
        peak = max(abs(s) for s in samples)
        bars = int(peak / 32768 * 40)
        print(f"\r  Level: {'█' * bars:40s} {peak:5d}", end="", flush=True)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Save
    wf = wave.open(TMP_FILE, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print(f"\n✅ Saved to {TMP_FILE}")
    return TMP_FILE


def test_playback(path: str):
    """Play back recorded audio."""
    import pyaudio

    wf = wave.open(path, "rb")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )

    print(f"\n🔈 Playing back {path}...")
    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    wf.close()
    print("✅ Playback done!")


def test_piper_tts():
    """Quick TTS test if Piper is installed."""
    import subprocess
    import shutil

    if not shutil.which("piper"):
        print("\n⚠️  Piper not installed yet, skipping TTS test")
        return

    out_path = "/tmp/nanobot/test_tts.wav"
    text = "Hello, I am NanoBot. My systems are operational."

    print(f"\n🗣️  Testing Piper TTS...")
    result = subprocess.run(
        ["piper", "--model", "models/piper/en_US-lessac-medium.onnx",
         "--output_file", out_path],
        input=text, capture_output=True, text=True,
    )

    if result.returncode == 0:
        print("✅ TTS synthesis successful!")
        test_playback(out_path)
    else:
        print(f"❌ TTS failed: {result.stderr}")


def main():
    print("=" * 50)
    print("  🤖 NanoBot Audio Test")
    print("=" * 50)

    try:
        import pyaudio
    except ImportError:
        print("❌ PyAudio not installed! Run: pip install pyaudio")
        sys.exit(1)

    list_devices()

    print("\n--- Test 1: Record & Playback ---")
    path = test_record()
    test_playback(path)

    print("\n--- Test 2: TTS (if available) ---")
    test_piper_tts()

    print("\n" + "=" * 50)
    print("  ✅ Audio test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
