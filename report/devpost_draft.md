# TinyLLM-ARM-Pro — Devpost Submission Draft

## Title
**TinyLLM-ARM-Pro — Hardware-Aware LLM Inference Optimization for Arm64**

## Tagline
1.1B parameters · One ARM chip · 5.75× faster — LLM inference for the other 6 billion.

## Project Description (what it does)

TinyLLM-ARM-Pro is a hardware-aware LLM inference optimizer for Arm64. It takes a
1.1B-parameter chat LLM, quantizes it with scientific rigor (K-quant mixed precision),
runs it through a runtime built from source for ARM, accelerates it with hand-written
NEON and I8MM SIMD kernels, and — in one command — detects the host chip, recommends
the optimal quantization, validates it with live inference, and reports an honest
prediction error. No GPU. No cloud. Just ARM.

## Built With
Python 3.14 · llama.cpp · PyTorch · Apple Metal · ARM NEON · I8MM (SMMLA) · Flash Attention · Three.js · GitHub Actions (Cobalt 100 ARM64)

---

## Story — mapped to the judging rubric

### 1. The Mission (Impact — 20%)
Most LLM inference runs on cloud GPUs that most of the world's developers will never
afford or access. TinyLLM-ARM-Pro is built for the other 6 billion: private, offline,
affordable LLM inference on the Arm64 silicon people already own — from Apple Silicon
laptops to cloud ARM64 instances and edge devices.

### 2. The Science (Technological Implementation — 40%)
Quantization was treated as an experiment, not a habit. Four quantization levels
(Q2_K, Q4_K_M, Q5_K_M, Q8_0) were benchmarked on the same model, workload and
hardware, with quality measured two ways: an academic WikiText-2 run (Q4_K_M: 8.73 PPL
vs the community reference 8.74 — within 0.14%) and a dev-time pseudo-perplexity
metric. The counter-intuitive winner — Q4_K_M, the smallest practical precision —
repeats across a second model family (Phi-2 2.7B), proving the pipeline is
model-agnostic.

### 3. The Low Level (Technological Implementation — 40%)
The project goes below the framework: hand-written ARM kernels in C
(`kernels/neon_gemm_v2.c`, `kernels/neon_i8mm.c`) using 128-bit NEON registers
(4-lane FP32 FMLA) and I8MM SMMLA (8×8 INT8 dot products) with pre-packed weight
layouts — the same instruction family Arm ships in its Compute Library. Results:
12.52× NEON FP32 speedup, 12.48× I8MM INT8 speedup. Native llama-bench with Flash
Attention reaches 1,329 tokens/sec prompt processing on Apple M4.

### 4. Portability (Impact / Ecosystem)
The identical pipeline was executed on two independent Arm64 machines: an Apple M4
development machine and a Cobalt 100 cloud ARM64 instance (GitHub Actions
ubuntu-24.04-arm). Same commands, same Q4_K_M recommendation, kernel correctness
verified on both. Write once, run on any Arm64.

### 5. The Smart Part (Technological Implementation + UX)
An Adaptive Inference Planner detects the chip, predicts performance before
inference, validates live, and reports the true error (exact errors are published
with every run in `results/auto_mode_result.json` — typically under 7% for speed
and RAM). Combined with a one-command orchestrator (`python run_all.py --auto`),
an interactive chatbot demo, and a zero-dependency animated Mission Control report,
the project is usable in 60 seconds by any developer.

## Key Numbers
- 5.75× decode speedup over FP32 (19.0 → 109.2 tokens/sec, Apple M4)
- 83% RAM reduction (4.10 GB → 0.71 GB)
- 0.42s model load (vs 3.0s FP32)
- 1,329 tokens/sec prompt processing (Flash Attention, native llama-bench)
- 12.52× NEON FP32 kernel speedup · 12.48× I8MM INT8 kernel speedup
- WikiText-2: 8.73 PPL vs 8.74 reference (0.14%)
- 6% typical planner prediction error (exact per-run errors published in `results/`)
- 77.4% decode retention at 88% context utilization
- 2 model families (TinyLlama 1.1B, Phi-2 2.7B) · 2 Arm64 devices (M4, Cobalt 100)

## How We Built It
- **Quantization:** llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` for Q2_K /
  Q4_K_M / Q5_K_M / Q8_0
- **Runtime:** llama-cpp-python 0.3.34 compiled from source on ARM64; Metal GPU
  full-layer offload
- **Kernels:** hand-written C (NEON FP32 GEMM with multi-accumulator blocking, I8MM
  SMMLA with tiled pre-packed weights), benchmarked via a custom harness on M4 and
  Cobalt 100
- **Pipeline:** planner (hardware detection + prediction), live validation,
  reporting; every benchmark script ships in the repo
- **Verification culture:** clean-machine install test (a fresh clone + setup on a
  blank machine), public dev log, corrected baseline published openly when a
  measurement bug was found

## Instructions to Run
```bash
git clone https://github.com/JagadeeshwaranCEO/tinyllm-arm-pro.git
cd tinyllm-arm-pro
pip install -r requirements.txt
python run_all.py --auto                 # detect → recommend → validate → report
python pipeline/chatbot.py --model Q4_K_M # interactive on-device chat
open report/mission_control.html         # animated mission report
```

## Additional Media
- **Video:** 3-min demo (see `report/video_script.md` runbook)
- **Mission Control:** `report/mission_control.html` — animated report, zero deps
- **Dashboard:** `report/dashboard.html` — Three.js 3D performance galaxy
- **Research report:** `report/research_report.md` — full methodology
- **Dev log:** `dev_log.md` — 27 days of honest engineering history
