#!/usr/bin/env bash
#
# NanoBot Setup Script — Jetson Nano
# Run once after flashing JetPack: sudo bash setup.sh
#
set -euo pipefail

NANOBOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${NANOBOT_DIR}/models"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[NanoBot]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Check we're on Jetson ---
check_platform() {
    if [ -f /etc/nv_tegra_release ]; then
        log "Detected Jetson platform ✓"
        cat /etc/nv_tegra_release
    else
        warn "Not running on Jetson — some features will be simulated"
    fi
}

# --- System setup ---
setup_system() {
    log "Updating system packages..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential cmake git \
        portaudio19-dev libsndfile1-dev \
        libasound2-dev alsa-utils \
        libopenblas-dev liblapack-dev \
        libatlas-base-dev gfortran \
        curl wget unzip \
        software-properties-common

    # Install Python 3.8 (Ubuntu 18.04 ships 3.6 which is EOL)
    if ! command -v python3.8 &>/dev/null; then
        log "Installing Python 3.8 via deadsnakes PPA..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update
    fi
    sudo apt-get install -y \
        python3.8 python3.8-venv python3.8-dev python3.8-distutils

    # Install pip for 3.8
    if ! python3.8 -m pip --version &>/dev/null; then
        curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.8
    fi

    log "Python 3.8 installed (system python3 left unchanged to avoid breaking apt)"

    # Create swap file (critical for 4GB Nano)
    if [ ! -f /swapfile ]; then
        log "Creating 4GB swap file..."
        sudo fallocate -l 4G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        log "Swap enabled ✓"
    else
        log "Swap already exists ✓"
    fi

    # Set to headless (multi-user target) if running desktop
    if systemctl get-default | grep -q graphical; then
        log "Switching to headless mode (saves ~300MB RAM)..."
        sudo systemctl set-default multi-user.target
        warn "Reboot required for headless mode to take effect"
    fi

    # Max performance mode
    log "Setting Jetson to MAX performance mode..."
    sudo nvpmodel -m 0 2>/dev/null || true
    sudo jetson_clocks 2>/dev/null || true

    # Short J48 jumper reminder
    warn "REMINDER: Make sure J48 jumper is shorted for barrel jack power!"
}

# --- Python environment ---
setup_python() {
    log "Setting up Python 3.8 virtual environment..."
    cd "${NANOBOT_DIR}"

    # Remove old 3.6 venv if present
    if [ -d venv ] && venv/bin/python --version 2>&1 | grep -q "3\.6"; then
        warn "Removing old Python 3.6 venv..."
        rm -rf venv
    fi

    python3.8 -m venv venv --system-site-packages
    source venv/bin/activate

    pip install --upgrade pip

    # Core dependencies
    pip install \
        pyyaml \
        aiohttp \
        webrtcvad \
        numpy

    # PyAudio (needs portaudio headers)
    pip install pyaudio

    log "Python environment ready ($(python --version)) ✓"
}

# --- Patch ggml for CUDA 10.2 / GCC 7.5 compatibility ---
# Jetson Nano ships with CUDA 10.2 and GCC 7.5 which need three fixes:
#   1. NEON vld1q_*_x2/x4 intrinsics missing on GCC < 8 aarch64
#   2. constexpr __device__ not supported in CUDA < 11
#   3. __builtin_assume not available in CUDA < 11.1
patch_ggml_cuda10() {
    local src_dir="$1"
    log "Applying CUDA 10.2 compatibility patches..."

    # Patch 1: Split NEON guard so vld1q_*_x2/x4 fallbacks are enabled on GCC < 8
    # The original code has a single #if !defined(__aarch64__) block covering:
    #   a) Functions that exist in GCC 7.5 (vmaxvq_f32, vzip1_u8, etc.) — must NOT redefine
    #   b) vld1q_*_x2/x4 wrappers missing in GCC < 8 — MUST provide fallbacks
    # We split (a) and (b) into separate guards.
    # File location differs: whisper.cpp uses ggml/src/, llama.cpp uses ggml/src/ggml-cpu/
    local cpu_impl=""
    if [ -f "${src_dir}/ggml/src/ggml-cpu-impl.h" ]; then
        cpu_impl="${src_dir}/ggml/src/ggml-cpu-impl.h"
    elif [ -f "${src_dir}/ggml/src/ggml-cpu/ggml-cpu-impl.h" ]; then
        cpu_impl="${src_dir}/ggml/src/ggml-cpu/ggml-cpu-impl.h"
    else
        warn "ggml-cpu-impl.h not found in ${src_dir}, skipping patch 1"
    fi
    if [ -n "$cpu_impl" ]; then
        # Insert #endif and new guard between vzip2_u8 block and vld1q_* block
        sed -i '/^\/\/ vld1q_s16_x2$/i\
#endif // !defined(__aarch64__)\
\
// vld1q_*_x2/x4 are missing on GCC < 8 even on aarch64\
#if !defined(__aarch64__) || (defined(__GNUC__) \&\& !defined(__clang__) \&\& __GNUC__ < 8)\
' "$cpu_impl"
    fi

    # Patch 2: Replace constexpr __device__ with const __device__ (CUDA 10.2 compat)
    sed -i 's/static constexpr __device__/static __device__ const/g' \
        "${src_dir}/ggml/src/ggml-cuda/common.cuh"

    # Patch 3: Replace bare __builtin_assume with GGML_CUDA_ASSUME macro (no-op on CUDA < 11.1)
    for f in "${src_dir}"/ggml/src/ggml-cuda/fattn-common.cuh \
             "${src_dir}"/ggml/src/ggml-cuda/fattn-vec-f16.cuh \
             "${src_dir}"/ggml/src/ggml-cuda/fattn-vec-f32.cuh; do
        [ -f "$f" ] && sed -i 's/__builtin_assume(/GGML_CUDA_ASSUME(/g' "$f"
    done

    log "CUDA 10.2 patches applied ✓"
}

# --- Build whisper.cpp with CUDA ---
# Pinned to v1.7.2: last release compatible with CUDA 10.2 (before C++17 became mandatory)
WHISPER_VERSION="v1.7.2"
setup_whisper() {
    if [ -x "${HOME}/whisper.cpp/build/bin/main" ]; then
        log "whisper.cpp already built, skipping (delete ~/whisper.cpp/build to rebuild)"
    else
        log "Building whisper.cpp ${WHISPER_VERSION} with CUDA support..."
        cd "${HOME}"

        if [ -d whisper.cpp ]; then
            cd whisper.cpp
            git fetch --tags
        else
            git clone https://github.com/ggerganov/whisper.cpp.git
            cd whisper.cpp
        fi
        git checkout "${WHISPER_VERSION}"

        patch_ggml_cuda10 "${HOME}/whisper.cpp"

        # Build with CUDA for Jetson Nano (sm_53)
        # CMAKE_CUDA_COMPILER_FORCED skips broken nvcc detection in CMake 3.25+ with CUDA 10.2
        mkdir -p build && cd build
        cmake .. \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCMAKE_CUDA_COMPILER_FORCED=TRUE \
            -DCMAKE_CUDA_ARCHITECTURES=53 \
            -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release -j$(nproc)

        # Symlink binary (v1.7.2 binary is named "main", not "whisper-cli")
        sudo ln -sf "${HOME}/whisper.cpp/build/bin/main" /usr/local/bin/whisper-cli
    fi

    # Download models
    mkdir -p "${MODELS_DIR}/whisper"
    cd "${HOME}/whisper.cpp"

    log "Downloading Whisper tiny.en model (English)..."
    bash models/download-ggml-model.sh tiny.en
    cp models/ggml-tiny.en.bin "${MODELS_DIR}/whisper/"

    log "Downloading Whisper tiny model (multilingual for Spanish)..."
    bash models/download-ggml-model.sh tiny
    cp models/ggml-tiny.bin "${MODELS_DIR}/whisper/"

    log "whisper.cpp ${WHISPER_VERSION} built and models downloaded ✓"
}

# --- Build llama.cpp with CUDA ---
# Pinned to b4262: last release compatible with CUDA 10.2 (before C++17 became mandatory)
LLAMA_VERSION="b4262"
setup_llama() {
    if [ -x "${HOME}/llama.cpp/build/bin/llama-server" ]; then
        log "llama.cpp already built, skipping (delete ~/llama.cpp/build to rebuild)"
    else
        log "Building llama.cpp ${LLAMA_VERSION} with CUDA support..."
        cd "${HOME}"

        if [ -d llama.cpp ]; then
            cd llama.cpp
            git fetch --tags
        else
            git clone https://github.com/ggerganov/llama.cpp.git
            cd llama.cpp
        fi
        git checkout "${LLAMA_VERSION}"

        patch_ggml_cuda10 "${HOME}/llama.cpp"

        mkdir -p build && cd build
        cmake .. \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCMAKE_CUDA_COMPILER_FORCED=TRUE \
            -DCMAKE_CUDA_ARCHITECTURES=53 \
            -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release -j$(nproc)

        sudo ln -sf "${HOME}/llama.cpp/build/bin/llama-server" /usr/local/bin/llama-server
        sudo ln -sf "${HOME}/llama.cpp/build/bin/llama-cli" /usr/local/bin/llama-cli
    fi

    # Download TinyLlama model
    mkdir -p "${MODELS_DIR}/llm"
    cd "${MODELS_DIR}/llm"

    if [ ! -f tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf ]; then
        log "Downloading TinyLlama 1.1B Chat Q4_K_M..."
        wget -q --show-progress \
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    fi

    log "llama.cpp ${LLAMA_VERSION} built and TinyLlama downloaded ✓"
}

# --- Install Piper TTS ---
setup_piper() {
    log "Installing Piper TTS..."
    cd "${HOME}"

    # Download pre-built Piper for aarch64
    PIPER_VERSION="2023.11.14-2"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"

    if [ ! -f piper/piper ]; then
        wget -q --show-progress -O piper.tar.gz "${PIPER_URL}"
        tar xzf piper.tar.gz
        rm piper.tar.gz
    fi

    sudo ln -sf "${HOME}/piper/piper" /usr/local/bin/piper

    # Download voice models
    mkdir -p "${MODELS_DIR}/piper"
    cd "${MODELS_DIR}/piper"

    # English voice
    if [ ! -f en_US-lessac-medium.onnx ]; then
        log "Downloading English voice (lessac-medium)..."
        wget -q --show-progress \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        wget -q --show-progress \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    fi

    # Spanish voice
    if [ ! -f es_ES-davefx-medium.onnx ]; then
        log "Downloading Spanish voice (davefx-medium)..."
        wget -q --show-progress \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
        wget -q --show-progress \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
    fi

    log "Piper TTS installed with EN + ES voices ✓"
}

# --- Install LED library ---
setup_leds() {
    log "Installing rpi_ws281x for Jetson..."
    cd "${NANOBOT_DIR}"
    source venv/bin/activate

    # The Jetson-compatible fork
    pip install rpi_ws281x adafruit-circuitpython-neopixel

    log "LED library installed ✓"
}

# --- Create systemd service ---
setup_service() {
    log "Creating systemd service..."

    sudo tee /etc/systemd/system/nanobot.service > /dev/null << EOF
[Unit]
Description=NanoBot Conversational Robot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${NANOBOT_DIR}
ExecStartPre=${HOME}/llama.cpp/build/bin/llama-server -m ${MODELS_DIR}/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8080 -ngl 24 -c 1024 &
ExecStart=${NANOBOT_DIR}/venv/bin/python ${NANOBOT_DIR}/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Separate service for llama.cpp server
    sudo tee /etc/systemd/system/llama-server.service > /dev/null << EOF
[Unit]
Description=llama.cpp Server for NanoBot
After=network.target

[Service]
Type=simple
User=root
ExecStart=${HOME}/llama.cpp/build/bin/llama-server \
    -m ${MODELS_DIR}/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl 24 -c 1024 --threads 4
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    log "Services created (enable with: sudo systemctl enable llama-server nanobot)"
}

# --- Create tmp directory ---
setup_dirs() {
    mkdir -p /tmp/nanobot
    mkdir -p "${MODELS_DIR}"/{whisper,llm,piper}
    log "Directories created ✓"
}

# --- Main ---
main() {
    echo ""
    echo "  🤖 NanoBot Setup Script"
    echo "  ========================"
    echo ""

    check_platform
    setup_dirs

    case "${1:-all}" in
        all)
            setup_system
            setup_python
            setup_whisper
            setup_llama
            setup_piper
            setup_leds
            setup_service
            ;;
        system)   setup_system ;;
        python)   setup_python ;;
        whisper)  setup_whisper ;;
        llama)    setup_llama ;;
        piper)    setup_piper ;;
        leds)     setup_leds ;;
        service)  setup_service ;;
        *)
            echo "Usage: sudo bash setup.sh [all|system|python|whisper|llama|piper|leds|service]"
            exit 1
            ;;
    esac

    echo ""
    log "============================================"
    log "  Setup complete! 🎉"
    log "============================================"
    echo ""
    log "Next steps:"
    log "  1. Reboot if switched to headless mode"
    log "  2. Test audio:    python scripts/test_audio.py"
    log "  3. Test LEDs:     sudo python scripts/test_leds.py"
    log "  4. Start server:  sudo systemctl start llama-server"
    log "  5. Run NanoBot:   sudo python main.py"
    echo ""
}

main "$@"
