# Contributing to TinyLLM-ARM-Pro

Thank you for your interest in contributing. This project welcomes contributions
that help make ARM AI inference more accessible to developers worldwide.

## Ways to Contribute

### 1. New Device Benchmarks
Run the benchmark suite on a device we have not yet tested and submit results:
- Raspberry Pi 4 / Pi 5
- AWS Graviton (any generation)
- Android device via Termux
- Any ARM64 Linux machine

Run: `python pipeline/run_all.py --quick`
Submit results as a PR adding your device's JSON to `results/devices/`

### 2. New Model Support
Add a new model family to the pipeline:
- Download model from HuggingFace
- Run through `quantize/build_and_quantize.sh`
- Add benchmark script following `benchmarks/phi2_benchmark.py` pattern
- Update `pipeline/planner.py` model registry

### 3. Bug Reports
Open an issue with:
- Your device and OS
- Exact command that failed
- Full error output
- Output of `python pipeline/run_all.py --detect-hardware`

### 4. Kernel Optimizations
NEON and I8MM kernels live in `kernels/`. All contributions must:
- Pass correctness checks against the naive baseline
- Show measured speedup on at least one ARM64 device
- Include the benchmark output in the PR description

## Setup

```bash
git clone https://github.com/JagadeeshwaranCEO/tinyllm-arm-pro
cd tinyllm-arm-pro
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Code Standards
- Python: follow existing style, no external linters required
- C kernels: must compile with `clang -O3 -march=native`
- Every benchmark script must save results to `results/*.json`
- Every claim must be backed by a reproducible measurement

## Project Mission

This project exists to prove that production-quality AI inference
is possible on any ARM device a developer already owns.
Contributions that extend this proof — to new devices, new models,
new optimizations — are the highest-value additions.

Built in Tamil Nadu. For the world.
