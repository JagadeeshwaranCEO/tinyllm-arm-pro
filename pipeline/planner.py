#!/usr/bin/env python3
"""
TinyLLM-ARM-Pro | Adaptive Inference Planner
═══════════════════════════════════════════════════════════
Given detected hardware + real measured benchmark data,
recommends the best quantization config, explains the
tradeoffs of every option (not just the winner), and
optionally validates the recommendation with a live run.

Design principle: this is pure decision logic, separate
from orchestration (run_all.py) and from benchmarking
(benchmarks/*.py). It only READS existing JSON results —
it never re-measures anything itself except when explicitly
asked to validate.
═══════════════════════════════════════════════════════════
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"

# Headroom rule: model RAM should stay under this fraction of
# total system RAM, leaving room for KV cache growth, OS overhead,
# and other running processes. See dev_log Day 21 for the reasoning.
RAM_HEADROOM_FRACTION = 0.40


def load_json(filename):
    """Load a results JSON file, return None if missing."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_benchmark_data():
    """
    Pulls together the two data sources the Planner needs:
    - leaderboard.json: speed + RAM per quantization level
    - accuracy.json: perplexity per quantization level (informational)

    Returns a merged dict keyed by quant name, or raises a clear
    error if the required data doesn't exist yet (rather than
    silently returning an empty/wrong recommendation).
    """
    leaderboard = load_json("leaderboard.json")
    accuracy = load_json("accuracy.json")

    if leaderboard is None:
        raise FileNotFoundError(
            "results/leaderboard.json not found. "
            "Run `python benchmarks/leaderboard.py` first — "
            "the Planner needs real measured data, not guesses."
        )

    accuracy_by_name = {}
    if accuracy is not None:
        for r in accuracy["results"]:
            accuracy_by_name[r["name"]] = r.get("perplexity")

    merged = {}
    for r in leaderboard["results"]:
        name = r["name"]
        merged[name] = {
            "name": name,
            "tokens_per_sec": r["tokens_per_sec"],
            "ram_gb": r["ram_gb"],
            "load_time": r["load_time"],
            "perplexity": accuracy_by_name.get(name),  # may be None
        }
    return merged


def recommend(total_ram_gb, benchmark_data):
    """
    Core decision logic. Pure function: same inputs always
    produce the same output — easy to test, easy to reason about.

    Returns a dict with:
      - recommendation: the chosen quant level's data
      - all_options: every candidate, annotated with fit/speed rank
      - warning: string if no option comfortably fits, else None
      - safe_limit_gb: the RAM threshold used for the decision
    """
    safe_limit = round(total_ram_gb * RAM_HEADROOM_FRACTION, 2)

    options = list(benchmark_data.values())
    fitting = [o for o in options if o["ram_gb"] <= safe_limit]

    warning = None
    if fitting:
        recommendation = max(fitting, key=lambda o: o["tokens_per_sec"])
    else:
        # Nothing fits comfortably — fall back to smallest, warn clearly
        recommendation = min(options, key=lambda o: o["ram_gb"])
        warning = (
            f"No quantization fits comfortably under the {safe_limit}GB "
            f"safety margin ({int(RAM_HEADROOM_FRACTION*100)}% of {total_ram_gb}GB RAM). "
            f"Falling back to the smallest available option "
            f"({recommendation['name']}, {recommendation['ram_gb']}GB). "
            f"Expect tight memory margins under load."
        )

    # Annotate every option for the explanation table —
    # this is what makes the output explainable, not a black box
    annotated = sorted(options, key=lambda o: -o["tokens_per_sec"])
    for o in annotated:
        o["fits_safely"] = o["ram_gb"] <= safe_limit
        o["is_recommended"] = (o["name"] == recommendation["name"])

    return {
        "total_ram_gb": total_ram_gb,
        "safe_limit_gb": safe_limit,
        "headroom_fraction": RAM_HEADROOM_FRACTION,
        "recommendation": recommendation,
        "all_options": annotated,
        "warning": warning,
    }


def format_explanation(plan):
    """
    Human-readable explanation of the decision — this is the
    'explain tradeoffs' step. Shows every option, not just the
    winner, so the recommendation isn't a black box.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ARM Adaptive Inference Planner")
    lines.append("=" * 60)
    lines.append(f"System RAM        : {plan['total_ram_gb']} GB")
    lines.append(
        f"Safety threshold   : {plan['safe_limit_gb']} GB "
        f"({int(plan['headroom_fraction']*100)}% of total RAM)"
    )
    lines.append("")
    lines.append(f"{'Quant':<10} {'Speed':>12} {'RAM':>8} {'Fits?':>7} {'Pick':>6}")
    lines.append("-" * 60)

    for o in plan["all_options"]:
        fit_mark = "✅" if o["fits_safely"] else "❌"
        pick_mark = "👈" if o["is_recommended"] else ""
        lines.append(
            f"{o['name']:<10} {o['tokens_per_sec']:>9.2f} t/s "
            f"{o['ram_gb']:>6.2f}GB {fit_mark:>7} {pick_mark:>6}"
        )

    lines.append("")
    rec = plan["recommendation"]
    lines.append(f"🏆 Recommended: {rec['name']}")
    lines.append(f"   Speed      : {rec['tokens_per_sec']} tok/s")
    lines.append(f"   RAM usage  : {rec['ram_gb']} GB")
    if rec.get("perplexity") is not None:
        lines.append(f"   Perplexity : {rec['perplexity']} (dev metric, see report for academic PPL)")

    if plan["warning"]:
        lines.append("")
        lines.append(f"⚠️  {plan['warning']}")
    else:
        lines.append("")
        lines.append(
            f"   Reason: fastest option that stays within the "
            f"{int(plan['headroom_fraction']*100)}% RAM safety margin."
        )

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    # Standalone test: uses a fake 17.2GB RAM (your M4) to verify
    # logic without needing the full pipeline. Real usage happens
    # through run_all.py --auto, which passes real detected hardware.
    print("Running planner.py standalone test (using 17.2GB test RAM)...\n")
    try:
        data = load_benchmark_data()
        plan = recommend(total_ram_gb=17.2, benchmark_data=data)
        print(format_explanation(plan))
    except FileNotFoundError as e:
        print(f"❌ {e}")