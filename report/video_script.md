# TinyLLM-ARM-Pro — Demo Video Runbook (Day 3)

**Target:** ~3 minutes · 1080p+ · for Devpost submission + GitHub README
**Judges watching:** Avin (reproducibility) · Michael (inference rigor) · Gabriel (ML science) · Rani (portability) · Disha (adoption) · Sicong (low-level Arm)

---

## Pre-flight checklist

- [ ] `git pull` on the submission repo — record from a **fresh clone** to prove reproducibility (Avin's question: "does it work on my machine?")
- [ ] Terminal: 42pt font minimum, dark theme, high contrast (it must read on a phone)
- [ ] Mission Control open in browser, pre-loaded, Replay button ready (`report/mission_control.html`)
- [ ] Dashboard open (`report/dashboard.html`)
- [ ] Record with QuickTime (Screen Recording, both monitor audio off) — or `ffmpeg -f avfoundation` if installed
- [ ] Phone-brightness test after each take: text must be readable

## Shot list

### S1 — HOOK · 0:00–0:20 (WOW + Avin)
Fresh clone in terminal. Type:
```bash
git clone https://github.com/JagadeeshwaranCEO/tinyllm-arm-pro.git
cd tinyllm-arm-pro && pip install -r requirements.txt
python run_all.py --auto
```
Overlay text as it runs: `DETECTING… Apple M4 · 17.2 GB · 10 cores` → `RECOMMENDING… Q4_K_M` → `VALIDATING… 108 tok/s live (3.4% error)`.
**VO:** *"One point one billion parameters. One ARM chip. One command. Watch the pipeline think."*

### S2 — THE MISSION · 0:20–0:45 (Disha + Impact)
Cut to Mission Control #mission section. Slow scroll through hero.
**VO:** *"Most LLMs live in cloud GPUs that six billion people will never touch. TinyLLM-ARM-Pro makes LLM inference run on the Arm64 chips people actually have — private, offline, and affordable. That's the mission."*

### S3 — THE STACK · 0:45–1:10 (Michael)
Mission Control #stack. Pause on Layer 4 (OUR KERNELS).
**VO:** *"The speed isn't luck — it's engineered at every layer: K-quant Q4_K_M chosen by evidence, llama.cpp built from source for Arm64, Metal offload — and at the bottom, kernels we wrote by hand."*

### S4 — THE SCIENCE · 1:10–1:35 (Gabriel + Michael)
Mission Control #surprise. Bars animate.
**VO:** *"Quantization is a science, not a hack. On academic WikiText-2, Q4_K_M scores 8.73 perplexity against the community's 8.74 reference — the same quality, at 5.75× the speed, in 83% less memory. And it beats INT8. The same verdict holds on Phi-2, a completely different model family."*

### S5 — THE LOW LEVEL · 1:35–2:00 (Sicong — the critical one)
Mission Control #lab. Click the I8MM tab during the take. Zoom (or re-record close-up) on the register animation.
**VO:** *"At the bottom we speak the chip's language — 128-bit NEON registers doing fused multiply-add across four lanes, and I8MM SMMLA doing 8-bit dot products, the same instruction family Arm ships in its Compute Library. Our kernels: 12.5× faster than scalar. Flash Attention prefill: 1,329 tokens a second."*

### S6 — PORTABILITY · 2:00–2:25 (Rani)
Mission Control #portable. Split-screen emphasis.
**VO:** *"The same pipeline ran on two different Arm64 machines — this MacBook's M4, and a Cobalt 100 cloud instance in a GitHub Actions datacenter. Same commands. Same Q4_K_M verdict. Write once, run on any Arm64."*

### S7 — THE PLANNER + CHATBOT · 2:25–2:45 (Michael + Disha)
Mission Control #planner, then live terminal:
```bash
python pipeline/chatbot.py --model Q4_K_M
> Explain quantization in simple terms
```
Show tokens streaming.
**VO:** *"The planner predicts your hardware's performance before running — then validates live and publishes the honest error. And when it's done benchmarking, you can just talk to the model."*

### S8 — CLOSE · 2:45–3:00
Mission Control #close. Pull back, let the metrics strip fill.
**VO:** *"TinyLLM-ARM-Pro. Hardware-aware inference optimization for Arm64, open source, MIT licensed, every number reproducible with one command. LLM inference for the other six billion."*

---

## Editing notes

- No need to re-record: S2–S6 are pure browser scrolls of Mission Control — record once, cut markers at section boundaries
- Cut on the bell: voiceover drives; any dead air longer than 0.6s gets trimmed
- Caption every number (judges may watch muted)
- End card: repo URL + `python run_all.py --auto` as the closing frame

## Fallback if recording fails

Screen record the Mission Control Replay button walkthrough (auto-scrolls all sections) and overlay VO — the same story, zero terminal footage needed.
