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
## Day 6 — June 16, 2026
- Built perplexity measurement pipeline across all quantization levels
- Results — Speed vs Accuracy tradeoff on ARM64:
  - Q2_K  : 78.28 tok/s  | perplexity 127.46 (+10.3% degradation)
  - Q4_K_M: 109.20 tok/s | perplexity 111.75 (-3.3% BETTER than Q8_0)
  - Q5_K_M: 71.53 tok/s  | perplexity 115.66 (+0.1%)
  - Q8_0  : 64.90 tok/s  | perplexity 115.57 (reference)
- Key finding: Q4_K_M beats Q8_0 in BOTH speed and accuracy on ARM Metal GPU
- This confirms INT4 K-quant aligns optimally with ARM SIMD vector widths
- Status: Full benchmark suite complete. Day 7 builds the README.
## Day 7 — June 16, 2026
- Built production-grade README with full benchmark tables
- Added badges: ARM64, Python, MIT, llama.cpp, Metal GPU
- Documented complete technical deep dive:
  - K-quant block structure explanation
  - ARM Metal GPU offload methodology
  - MLPerf benchmark methodology
  - Full reproduction instructions
- Added "What's Next" roadmap for judges
- README renders live on GitHub front page
- Status: Documentation complete. Day 8 is demo video.

## Day 8 — June 16, 2026
- Built llama.cpp v9660 from source on ARM64 (AppleClang 17)
- Confirmed ARM64 binary: NEON, I8MM, dotprod, Metal all enabled
- Built own quantization pipeline from scratch
- Quantized TinyLlama 1.1B ourselves (not downloaded):
  - F16 source  : 2.0GB  (16.00 BPW)
  - Q2_K (ours) : 412MB  (3.14 BPW) — 80% compression
  - Q4_K_M (ours): 637MB (4.85 BPW) — 69% compression
  - Q8_0 (ours) : 1.1GB  (8.50 BPW) — 45% compression
- Status: Own quantization pipeline complete. Day 9 benchmarks our models.