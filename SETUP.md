# TinyLLM-ARM-Pro | Setup Guide

Step-by-step instructions for macOS and Linux ARM64.

---

## Requirements

- **Hardware:** Any ARM64 device (Apple Silicon M1–M4, AWS Graviton, Raspberry Pi 4/5)
- **OS:** macOS 13+ (Ventura) or Linux ARM64 (Ubuntu 22.04+, Debian 12+)
- **Python:** 3.11+
- **RAM:** 8 GB minimum (16 GB recommended)
- **Disk:** ~5 GB free for models

---

## macOS Setup

### 1. Clone and Environment

```bash
git clone https://github.com/JagadeeshwaranCEO/tinyllm-arm-pro.git
cd tinyllm-arm-pro
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install llama.cpp from Source (Recommended)

Building from source with `GGML_NATIVE=ON` enables chip-specific optimizations:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_METAL=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
cd ..
```

Verify ARM capabilities:
```bash
./llama.cpp/build/bin/llama-quantize --help 2>&1 | head -5
```

### 4. Download Models

```bash
# GGUF quantized variants from HuggingFace
pip install huggingface_hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf \
  tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --local-dir ./models/gguf
```

### 5. Verify Setup

```bash
# Quick sanity check — should print hardware info and recommend Q4_K_M
python pipeline/run_all.py --detect-hardware

# Run the Adaptive Inference Planner
python pipeline/run_all.py --auto

# Full benchmark suite
python pipeline/run_all.py

# Interactive chatbot demo
python pipeline/chatbot.py
```

---

## Linux ARM64 Setup (Ubuntu/Debian)

### 1. System Dependencies

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake curl \
  python3 python3-pip python3-venv \
  git ninja-build
```

### 2. Clone and Python

```bash
git clone https://github.com/JagadeeshwaranCEO/tinyllm-arm-pro.git
cd tinyllm-arm-pro
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=OFF
cmake --build build --config Release -j$(nproc)
cd ..
```

> **Note:** On Linux, Metal GPU is not available. The model runs on CPU with NEON-optimized
> inference. For GPU acceleration on ARM Linux, consider OpenCL or Vulkan backends.

### 4. Download Models and Run

```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q2_K.gguf \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf \
  tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --local-dir ./models/gguf

python pipeline/run_all.py --detect-hardware
```

---

## Quantization from Source (Advanced)

To quantize models yourself rather than downloading pre-quantized GGUF files:

```bash
# Convert HuggingFace model to GGUF FP16
python llama.cpp/convert_hf_to_gguf.py ./models/tinyllama --outtype f16 \
  --output ./models/own_quantized/tinyllama-f16.gguf

# Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize \
  ./models/own_quantized/tinyllama-f16.gguf \
  ./models/own_quantized/tinyllama-own-q4km.gguf \
  Q4_K_M

# Use the automated script
bash quantize/build_and_quantize.sh
```

---

## Running Benchmarks

```bash
# FP32 baseline (requires PyTorch MPS)
python benchmarks/baseline.py

# Single Q4_K_M benchmark
python benchmarks/quant_benchmark.py

# Full multi-quant leaderboard
python benchmarks/leaderboard.py

# Accuracy + perplexity analysis
python benchmarks/accuracy.py

# Pipeline validation (ours vs reference)
python benchmarks/pipeline_validation.py

# Cross-model comparison (TinyLlama vs Phi-2)
python benchmarks/cross_model.py

# Context scaling stress test
python benchmarks/stress_test.py

# Native llama-bench (requires llama.cpp built from source)
bash benchmarks/native_benchmark.sh
```

---

## View Dashboard

Open `report/dashboard.html` in any browser. The dashboard fetches live data
from `results/master_results.json`. When opened via `file://` protocol, it
uses a built-in fallback snapshot.

---

## Troubleshooting

### `pip install` fails for llama-cpp-python

Install from pre-built wheels:
```bash
CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python --no-cache-dir
```

### "No module named 'llama_cpp'"

Ensure you're in the activated virtual environment:
```bash
which python  # should show /path/to/tinyllm-arm-pro/venv/bin/python
source venv/bin/activate  # if not
```

### Model file not found

Verify model files exist:
```bash
ls -la models/gguf/
```
Expected: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` etc.

### Dashboard shows fallback/cached snapshot

This is expected when opening via `file://` (browser security restrictions prevent
fetch from local filesystem). Serve with a local HTTP server:
```bash
python -m http.server 8080
# Open http://localhost:8080/report/dashboard.html
```

---

*Built for the ARM Create: AI Optimization Challenge 2026*
