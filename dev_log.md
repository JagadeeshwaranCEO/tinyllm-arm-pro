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
## Day 9 — June 17, 2026
- Ran full pipeline validation: Our models vs TheBloke reference
- Converted FP32 → GGUF F16 → Q4_K_M/Q8_0/Q2_K in correct project directory
- Results — Our Pipeline vs Reference:

  Q4_K_M:
  - Speed      : 107.55 vs 109.50 tok/s  (-1.8% — essentially equal)
  - Perplexity : 29.16  vs 111.75        (-73.9% — our model dramatically better)
  - RAM        : 0.71GB vs 0.65GB        (comparable)

  Q8_0:
  - Speed      : 75.51  vs 73.02 tok/s   (+3.4% — we're faster)
  - Perplexity : 32.33  vs 115.57        (-72.0% — our model dramatically better)

  Q2_K:
  - Speed      : 104.89 vs 81.72 tok/s   (+28.4% — we're significantly faster)
  - Perplexity : 50.72  vs 127.46        (-60.2% — our model dramatically better)

- Key Finding: Our pipeline produces superior quality AND equal/better speed
- Results saved to results/pipeline_validation.json
- Status: Pipeline validated. Day 10 builds automated results export.
## Day 10 — June 17, 2026
- Built run_all.py — single command master pipeline
- Auto hardware detection: detected Apple M4, recommended Q4_K_M
- Pipeline runs: quantization → baseline → leaderboard → validation → report
- Master results aggregated to results/master_results.json
- Summary prints mission statement with all key findings
- 3/4 steps passed on first run — fixed bash/python script routing
- Status: Master pipeline operational. Day 11 builds live dashboard.

## Day 11 — June 17, 2026
- Rebuilt 3D deep space dashboard with live data binding
- Dashboard fetches results/master_results.json via JS fetch()
- Fallback snapshot included for offline/file:// viewing
- Auto-renders: leaderboard bars, perplexity comparison cards, metric grid
- Confirmed "live data connected" status — real numbers, zero hardcoding
- Tested via local server: python3 -m http.server 8000
- Status: Dashboard fully automated. Day 12 starts NEON kernel work.

## Day 12 (continued) — June 18, 2026
- Built NEON v2: 4-row blocking + 4 independent accumulators
- Breaks single FMA dependency chain from v1, improves cache reuse
- Results (naive vs v1 vs v2):
  - 64x64x64      : v1 4.96x -> v2 12.12x  (2.45x improvement)
  - 256x256x256   : v1 3.63x -> v2 11.19x  (3.08x improvement)
  - 512x512x512   : v1 3.57x -> v2 12.10x  (3.38x improvement)
  - 2048x64x2048  : v1 3.91x -> v2 12.52x  (3.20x improvement)
- All results verified correct against naive baseline
- Key insight: M4's NEON pipeline rewards ILP more than predicted —
  multiple independent accumulators matter more than raw SIMD width
- Status: NEON kernel v2 complete. Best result: 12.52x speedup, verified.

## Day 13 — June 18, 2026
- Ran native llama-bench with full optimization flags:
  -ngl 99 (all layers Metal GPU)
  -fa 1 (Flash Attention enabled)
  -b 2048 -ub 2048 (Apple-recommended batch size)
  --cache-type-k/v q8_0 (quantized KV cache)

- Native benchmark results (pp512 / tg128):
  Q4_K_M : 1329.34 t/s prompt | 123.98 t/s generation
  Q8_0   : 1384.69 t/s prompt |  77.45 t/s generation
  Q2_K   : 1300.75 t/s prompt | 130.41 t/s generation

- Previous Python wrapper measured ~109 t/s — native is 12x higher on prompt
- Flash Attention accounts for massive pp improvement (attention is O(n²) without it)
- Status: Full native benchmark suite complete. Updating README with real numbers.
## Day 13 (continued) — June 18, 2026
- I8MM (SMMLA vmmlaq_s32) kernel — verified correct on all sizes
- Performance results:
  - 64x64    : 1.54x faster (instruction overhead manageable at small size)
  - 256x256+ : slower due to runtime memcpy for interleaving
- Root cause: vmmlaq_s32 requires pre-packed interleaved weight layout
  Our test uses row-major — runtime packing overhead dominates
- This is exactly why llama.cpp Q4_K_M pre-packs weights at quantization time
- Proved: vmmlaq_s32 works correctly on M4, correct future direction identified
- Full 4-layer optimization story now complete and documented honestly
## Day 13 (final) — June 18, 2026
- Fixed I8MM kernel with pre-packed B matrix layout
- Root cause was runtime memcpy interleaving killing performance
- Solution: pack_b_i8mm() called once at init (like llama.cpp at quant time)
- Fixed results:
  - 64x64     : 6.76x speedup
  - 256x256   : 2.57x speedup
  - 512x512   : 2.02x speedup
  - 2048x2048 : 3.08x speedup
- All correctness: PASSED ✅
- Full kernel story: naive → NEON v1 → NEON v2 → I8MM packed
- Status: All kernels complete and verified. Day 14 = research report.

## Day 14 — June 18, 2026
- Wrote complete research report (report/research_report.md)
- 8 pages covering:
  - Abstract with all key numbers
  - Background on NEON, I8MM, K-Quant
  - Full methodology (build environment, benchmark setup)
  - Results: quantization leaderboard, pipeline validation,
    native llama-bench, NEON kernels, I8MM kernels
  - Discussion: why native compilation matters, Q4_K_M sweet spot
  - Future work section
  - Conclusion with mission statement
- All numbers sourced from real benchmark outputs
- Status: Research report complete. Day 15 = demo video.

## Day 15 — June 19, 2026
- Added Limitations and Honest Scope section to research report
- Explicitly clarified: did not invent quantization formats, used existing GGUF
- Clarified 73.9% finding is a controlled comparison, not a categorical claim
- Clarified ARM does not beat GPU throughput — different value proposition
- Clarified I8MM kernel is educational, not production-optimized
- Clarified perplexity methodology is pseudo-PPL, not standard WikiText-2/C4
- Clarified hardware validation scope: M4 only, other devices are reasoning not measurement
- Status: Report is now fully honest and defensible. Day 16 = demo video script.
## Day 16 — June 19, 2026
- Downloaded official WikiText-2 raw test set
- Ran llama.cpp's official llama-perplexity tool on all 3 quantization levels
- Real academic-standard results (wiki.test.raw, 655 chunks):
  - Q8_0   : PPL 8.4459 ± 0.0533
  - Q4_K_M : PPL 8.7281 ± 0.0544
  - Q2_K   : PPL 12.3611 ± 0.0796
- Results follow expected academic pattern — directly citable, comparable to literature
- Updated research report Limitations section and added Section 4.1.3
- Status: One major limitation genuinely fixed with real measurement, not just rewording
- Next: multi-reference comparison (Day 17), I8MM multi-tile blocking (Day 18)

## Day 17 — June 19, 2026
- Downloaded second independent Q4_K_M source (andrijdavid/TinyLlama-1.1B-Chat-v1.0-GGUF)
- Ran identical WikiText-2 perplexity test for direct comparison
- Result: Ours 8.7281 vs Reference 8.7400 — statistically equivalent (within margin of error)
- IMPORTANT FINDING: corrects earlier 73.9% pseudo-perplexity claim
- Conclusion: our pipeline correctly reproduces expected Q4_K_M quality —
  the real contribution is a correct, from-scratch, verified pipeline,
  not an unusual quality advantage
- Updated research report abstract and Section 4.2.1 with honest correction
- This is the right scientific outcome — multiple independent Q4_K_M builds
  SHOULD converge since it's a deterministic algorithm
- Status: Report now scientifically rigorous and externally validated

## Day 18 — June 19, 2026
- Added i8mm_matmul_tiled: 2x8 output tile, 4 independent accumulators
- Same dependency-breaking principle as NEON v2, applied to INT8 SMMLA
- All correctness checks PASSED across all sizes
- Final I8MM kernel progression (naive -> packed -> tiled):
  - 64x64     : 6.76x -> 12.48x  (1.73x further improvement)
  - 256x256   : 2.55x -> 4.89x   (1.92x further improvement)
  - 512x512   : 2.03x -> 4.62x   (2.28x further improvement)
  - 2048x2048 : 3.59x -> 5.63x   (1.57x further improvement)
- Confirms: instruction-level parallelism via multiple accumulators
  is a general M4 optimization pattern, not specific to FP32 NEON
- Status: I8MM kernel fully optimized and verified. Day 19 = update research report.

## Day 19 — June 19, 2026
- Updated research report Section 4.5 with complete I8MM kernel evolution
- Documented v1 (broken) -> v2 (packed, fixed) -> v3 (tiled, optimized)
- Added cross-kernel insight: ILP principle generalizes from FP32 to INT8 SMMLA
- Report now tells complete, honest engineering investigation story
- Status: Research report fully complete and rigorous. Day 20 = demo video script.

## Day 20 — June 26, 2026
- Fixed leaderboard.py: now saves results/leaderboard.json + appends to history.jsonl
- Fixed accuracy.py: now saves results/accuracy.json + appends to history.jsonl
- Labeled accuracy.py output clearly as pseudo-perplexity dev metric (not academic)
- Created results/wikitext2_perplexity.json from real Day 16-17 measurements
  (was previously only in markdown report, now machine-readable)
- Both fixes are prerequisites for the Adaptive Inference Planner (Day 22-26)
  which needs real JSON data to make recommendations from
- Status: Benchmark plumbing fixed. Next: native_benchmark.sh JSON output, then Planner.

## Day 20 (continued) — June 26, 2026
- Fixed native_benchmark.sh: removed hardcoded ~/tinyllm-arm-pro/tinyllm-arm-pro paths,
  now resolves paths relative to script location (works on any machine/clone)
- Added benchmarks/_parse_native_benchmark.py: parses llama-bench table output into JSON
- All three benchmark sources now produce structured, consistent JSON:
  results/leaderboard.json, results/accuracy.json, results/native_benchmark.json
- All three append to results/history.jsonl — growing dataset for the Planner
- Status: All benchmark plumbing fixed and verified end-to-end. 
  Foundation ready for Adaptive Inference Planner (Day 21-26).
  ## Day 21 — June 26, 2026
- Designed and built pipeline/planner.py — the Adaptive Inference Planner core logic
- Pure decision function: recommend(total_ram_gb, benchmark_data) -> plan
- Logic: 40% RAM headroom rule (model RAM <= 40% of total RAM), pick fastest
  among options that fit; falls back to smallest option + explicit warning
  if nothing fits comfortably
- Output includes ALL options annotated (fits/recommended), not just the winner
  -> explainable, not a black box, per the design requirement
- Tested on two cases:
  - 17.2GB (M4 Mac): correctly recommends Q4_K_M, no warning
  - 1.0GB (simulated constrained device): correctly falls back to Q2_K
    with clear warning about tight memory margins
- Reads only existing JSON (leaderboard.json, accuracy.json) — does not
  re-measure anything itself, keeping decision logic separate from
  benchmarking and orchestration
- Status: Planner core logic complete and verified. Day 22 wires this
  into run_all.py --auto and adds the live validation step (step 4 of 4).

  ## Day 21 (stress test) — June 26, 2026
- Built context-scaling stress test on the Planner's recommended model (Q4_K_M)
- v1 flaw: subtraction-based timing (two model() calls) doubled measurement
  noise, producing impossible results (decode speed INCREASING 3x with context)
- Diagnosed root cause: llama.cpp's automatic prompt-prefix KV caching made
  the second calibration call skip most of its prefill, corrupting the subtraction
- v2 fix: direct decode-loop timing via model.eval() + model.scores argmax,
  3 repeats averaged per context level (matching llama-bench's -r rigor)
- Clean result: decode speed declines monotonically and smoothly,
  91.69 -> 67.37 tok/s (26.5% slowdown) from 6% to 88% context utilization
- Tight stdevs (±0.25 to ±3.45) confirm low noise, high confidence
- Caught and fixed a stale RAM note left over from v1's different caching behavior
- Status: First real workload stress test complete, methodology sound,
  result is publishable. This becomes Section 4.6 of the research report.