"""
report.py
---------
Reads results/results.jsonl and prints a human-readable analysis report.

Usage
-----
    python report.py                          # full report
    python report.py --category returns       # filter by category
    python report.py --failures-only          # show only failed cases
    python report.py --show-output            # include model output in report

What it shows
-------------
  - Overall pass rate
  - Pass rate by category
  - Per-scorer average scores
  - All failed cases with the scores that caused failure
  - LLM judge reasoning for each failure (most useful for debugging)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(path: str, category: str = None, failures_only: bool = False):
    results = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            if category and r.get("category") != category:
                continue
            if failures_only and r.get("passed"):
                continue
            results.append(r)
    return results


def print_report(results: list, show_output: bool = False):
    if not results:
        print("No results to report.")
        return

    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    pass_pct = passed / total * 100

    print(f"\n{'═'*60}")
    print(f"  ECHO-EVAL REPORT")
    print(f"{'═'*60}")
    print(f"  Total cases : {total}")
    print(f"  Passed      : {passed}  ({pass_pct:.0f}%)")
    print(f"  Failed      : {failed}")

    # ── By category ──────────────────────────────────────────
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\n{'─'*60}")
    print("  PASS RATE BY CATEGORY")
    print(f"{'─'*60}")
    for cat, cat_results in sorted(by_cat.items()):
        cat_passed = sum(1 for r in cat_results if r["passed"])
        cat_pct    = cat_passed / len(cat_results) * 100
        bar        = "█" * int(cat_pct / 5) + "░" * (20 - int(cat_pct / 5))
        print(f"  {cat:<18} {bar}  {cat_passed}/{len(cat_results)}  ({cat_pct:.0f}%)")

    # ── Average scores per scorer ─────────────────────────────
    scorer_keys = ["keyword_presence", "keyword_absence", "llm_judge_score", "refusal_detected"]
    print(f"\n{'─'*60}")
    print("  AVERAGE SCORES BY SCORER")
    print(f"{'─'*60}")
    for key in scorer_keys:
        values = [r["scores"][key] for r in results if key in r.get("scores", {})]
        if values:
            avg = sum(values) / len(values)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"  {key:<22} {bar}  {avg:.2f}  (n={len(values)})")

    # ── Failed cases detail ───────────────────────────────────
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{'─'*60}")
        print(f"  FAILED CASES ({len(failures)})")
        print(f"{'─'*60}")
        for r in failures:
            scores = r.get("scores", {})
            print(f"\n  ▸ [{r['case_id']}]  category={r['category']}  tags={r['tags']}")

            # Show which scores failed their thresholds
            thresholds = {
                "keyword_presence": 1.0,
                "keyword_absence": 1.0,
                "llm_judge_score": 0.7,
                "refusal_detected": 1.0
            }
            for key, threshold in thresholds.items():
                if key in scores and isinstance(scores[key], float):
                    val = scores[key]
                    status = "✓" if val >= threshold else "✗"
                    print(f"    {status} {key:<22} {val:.2f}  (threshold: {threshold})")

            # Show LLM judge reasoning if available
            if "llm_judge_reasoning" in scores:
                print(f"    → Judge: {scores['llm_judge_reasoning']}")
            if scores.get("llm_judge_gaps"):
                for gap in scores["llm_judge_gaps"]:
                    print(f"      · {gap}")

            # Show keyword violations
            if scores.get("keyword_presence_missing"):
                print(f"    → Missing keywords: {scores['keyword_presence_missing']}")
            if scores.get("keyword_absence_violations"):
                print(f"    → Forbidden keywords found: {scores['keyword_absence_violations']}")
            if scores.get("refusal_note"):
                print(f"    → Refusal note: {scores['refusal_note']}")

            # Optionally show the model output
            if show_output and r.get("model_output"):
                print(f"\n    Model output:")
                for line in r["model_output"].split("\n"):
                    print(f"      {line}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="echo-eval: report generator")
    parser.add_argument("--input",         default="results/results.jsonl")
    parser.add_argument("--category",      help="Filter by category")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--show-output",   action="store_true", help="Include model output in report")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"No results file found at {args.input}. Run runner.py first.")
        raise SystemExit(1)

    results = load_results(args.input, category=args.category, failures_only=args.failures_only)
    print_report(results, show_output=args.show_output)
