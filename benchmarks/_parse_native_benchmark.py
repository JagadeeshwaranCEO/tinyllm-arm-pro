#!/usr/bin/env python3
"""
Parses raw llama-bench text output (captured by native_benchmark.sh)
into structured JSON matching the shape used by leaderboard.json and
accuracy.json — so the Adaptive Inference Planner can read all three
benchmark sources consistently.

Why parse text instead of llama-bench's own JSON output flag: we've
already empirically confirmed the exact markdown table format this
build produces (see Day 13 logs), so parsing that known-good format
is more reliable than assuming a JSON schema we haven't verified
against this specific llama.cpp build.
"""
import argparse
import json
import os
import re
from datetime import datetime


def parse_log(path):
    """Extract pp512 / tg128 tokens-per-sec from a llama-bench table."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        text = f.read()
    for line in text.splitlines():
        if "pp512" not in line and "tg128" not in line:
            continue
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        test_name = next((c for c in cols if c in ("pp512", "tg128")), None)
        if not test_name:
            continue
        ts_field = cols[-1]  # e.g. "1329.34 ± 7.88"
        match = re.search(r"[-+]?\d+(\.\d+)?", ts_field)
        if not match:
            continue
        rows[test_name] = float(match.group())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args()

    results = []
    for label in args.labels:
        log_path = os.path.join(args.results_dir, f".tmp_native_{label}.txt")
        parsed = parse_log(log_path)
        if not parsed:
            print(f"⚠️  No data parsed for {label} (missing model or skipped run)")
            continue
        entry = {
            "name": label,
            "pp512_tokens_per_sec": parsed.get("pp512"),
            "tg128_tokens_per_sec": parsed.get("tg128"),
        }
        results.append(entry)
        print(f"✅ Parsed {label}: pp512={entry['pp512_tokens_per_sec']}  tg128={entry['tg128_tokens_per_sec']}")

    output = {
        "generated_at": datetime.now().isoformat(),
        "methodology": "llama-bench native (Flash Attention, Metal GPU, q8_0 KV cache)",
        "flags": "-ngl 99 -fa 1 -b 2048 -ub 2048 --cache-type-k q8_0 --cache-type-v q8_0 -p 512 -n 128 -r 5",
        "results": results,
    }

    out_path = os.path.join(args.results_dir, "native_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved: {out_path}")

    history_path = os.path.join(args.results_dir, "history.jsonl")
    with open(history_path, "a") as f:
        f.write(json.dumps(output) + "\n")
    print(f"✅ Appended to: {history_path}")

    # Clean up temp logs now that they're parsed
    for label in args.labels:
        tmp = os.path.join(args.results_dir, f".tmp_native_{label}.txt")
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()