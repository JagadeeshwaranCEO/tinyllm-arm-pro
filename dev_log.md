# TinyLLM-ARM-Pro Dev Log

## Day 1 — June 15, 2026
- Confirmed ARM64 architecture on Apple Silicon Mac
- Created GitHub repo: JagadeeshwaranCEO/tinyllm-arm-pro
- Set up Python 3.14.3 virtual environment
- Created full project structure
- Fought through git merge conflicts — fixed it
- Status: Environment ready. Day 2 starts inference work.
## Day 2 — June 15, 2026
- Downloaded TinyLlama 1.1B (2.2GB) locally
- Fixed Python 3.14 / httpx network bug by loading from local path
- First successful inference on Apple Silicon MPS
- FP32 baseline: 16.52 tokens/sec, model load 3.86s
- Status: Baseline established. Day 3 starts quantization.
## Day 3 — June 15, 2026
- Installed llama.cpp via llama-cpp-python
- Downloaded TinyLlama Q4_K_M GGUF (669MB vs 2.2GB FP32)
- Ran quantization benchmark on Apple Silicon Metal GPU
- Results:
  - FP32 baseline : 16.52 tokens/sec, 2.2GB RAM
  - Q4_K_M result : 96.36 tokens/sec, 0.76GB RAM
  - Speedup       : 5.83x faster
  - RAM reduction : 65% less memory
- Status: Core result achieved. Day 4 adds more quantization levels.