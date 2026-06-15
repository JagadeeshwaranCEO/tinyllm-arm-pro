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
## Day 4 — June 16, 2026
- Downloaded Q2_K, Q5_K_M, Q8_0 GGUF variants
- Built automated multi-model leaderboard benchmark
- Results on Apple Silicon ARM64:
  - FP32     : 16.52 tok/s (baseline)
  - Q2_K     : 81.03 tok/s (4.90x)
  - Q4_K_M   : 102.27 tok/s (6.19x) ← winner
  - Q5_K_M   : 80.71 tok/s (4.89x)
  - Q8_0     : 75.88 tok/s (4.59x)
- Key finding: Q4_K_M is optimal — fastest AND memory efficient
- Status: Leaderboard complete. Day 5 builds the report dashboard.
## Day 5 — June 16, 2026
- Built interactive 3D deep space performance dashboard using Three.js
- Features: draggable galaxy, orbital rings, animated counter, scroll parallax
- Benchmark results displayed in glassmorphism panels over the 3D scene
- Dashboard file: report/dashboard.html
- Status: Visual layer complete. Day 6 starts perplexity/accuracy measurement.