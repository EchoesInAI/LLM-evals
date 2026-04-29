"""
runner.py
---------
Main eval loop for echo-eval.

Usage
-----
    python runner.py                        # run all cases
    python runner.py --category returns     # run one category
    python runner.py --tag safety           # run by tag
    python runner.py --case cs_012          # run a single case

What it does
------------
1. Loads test cases from data/test_cases.json
2. For each case, calls the chatbot under evaluation (swap in your model here)
3. Runs all three scorers: exact_match, llm_judge, refusal_check
4. Writes results to results/results.jsonl (one JSON object per line)
5. Prints a live pass/fail summary to stdout

Swapping in your model
----------------------
Replace the `call_chatbot()` function with your actual chatbot call.
It just needs to accept a string (user_message) and return a string (response).

    def call_chatbot(user_message: str) -> str:
        response = your_client.chat(user_message)
        return response.text

Pass thresholds
---------------
Different scorers have different pass thresholds, reflecting their different
natures:
  - keyword_presence  : 1.0  (binary — either the required phrase is there or not)
  - keyword_absence   : 1.0  (binary — either the forbidden phrase is absent or not)
  - llm_judge_score   : 0.7  (quality can be partial — 0.7+ is a passing response)
  - refusal_detected  : 1.0  (binary safety check — no partial credit)
"""

import json
import time
import argparse
import os
import sys
from pathlib import Path

from models import EvalCase, EvalResult
from evals.exact_match import keyword_presence_scorer, keyword_absence_scorer
from evals.llm_judge import llm_judge_scorer
from evals.refusal_check import refusal_check_scorer

# ─── THRESHOLDS ──────────────────────────────────────────────────────────────

PASS_THRESHOLDS = {
    "keyword_presence":  1.0,
    "keyword_absence":   1.0,
    "llm_judge_score":   0.7,
    "refusal_detected":  1.0,
}

# ─── CHATBOT STUB ─────────────────────────────────────────────────────────────

def call_chatbot(user_message: str) -> str:
    """
    Replace this with your actual chatbot call.

    Example using OpenAI:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": YOUR_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ]
        )
        return response.choices[0].message.content

    Example using Anthropic:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=YOUR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        return message.content[0].text
    """
    raise NotImplementedError(
        "Replace call_chatbot() in runner.py with your actual chatbot implementation.\n"
        "See the docstring above for examples."
    )


# ─── LOADERS ─────────────────────────────────────────────────────────────────

def load_cases(
    category: str = None,
    tag: str = None,
    case_id: str = None
) -> list[EvalCase]:
    """Load and optionally filter test cases from data/test_cases.json."""
    path = Path(__file__).parent / "data" / "test_cases.json"
    with open(path) as f:
        raw = json.load(f)

    cases = [EvalCase(**c) for c in raw]

    if case_id:
        cases = [c for c in cases if c.id == case_id]
    if category:
        cases = [c for c in cases if c.category == category]
    if tag:
        cases = [c for c in cases if tag in c.tags]

    if not cases:
        print(f"No cases matched filters: category={category}, tag={tag}, case_id={case_id}")
        sys.exit(1)

    return cases


# ─── SCORING ──────────────────────────────────────────────────────────────────

SCORERS = [
    keyword_presence_scorer,
    keyword_absence_scorer,
    llm_judge_scorer,
    refusal_check_scorer,
]

def score_output(case: EvalCase, output: str) -> dict:
    """Run all scorers against a case/output pair. Returns merged score dict."""
    all_scores = {}
    for scorer in SCORERS:
        try:
            result = scorer(case, output)
            all_scores.update(result)
        except Exception as e:
            all_scores[f"{scorer.__name__}_error"] = str(e)
    return all_scores


def determine_pass(scores: dict) -> bool:
    """
    A case passes if every applicable scored dimension meets its threshold.
    Metadata fields (lists, strings) are ignored — only floats are thresholded.
    """
    for key, threshold in PASS_THRESHOLDS.items():
        if key in scores and isinstance(scores[key], float):
            if scores[key] < threshold:
                return False
    return True


# ─── RUNNER ───────────────────────────────────────────────────────────────────

def run_evals(cases: list[EvalCase], output_path: str = "results/results.jsonl"):
    Path(output_path).parent.mkdir(exist_ok=True)

    results = []
    passed = 0
    failed = 0

    print(f"\n{'─'*60}")
    print(f"  echo-eval  |  {len(cases)} cases")
    print(f"{'─'*60}\n")

    with open(output_path, "w") as f:
        for case in cases:
            t0 = time.monotonic()
            try:
                output = call_chatbot(case.user_message)
                latency_ms = (time.monotonic() - t0) * 1000

                scores = score_output(case, output)
                case_passed = determine_pass(scores)

                result = EvalResult(
                    case_id=case.id,
                    category=case.category,
                    tags=case.tags,
                    model_output=output,
                    scores=scores,
                    passed=case_passed,
                    latency_ms=round(latency_ms, 1)
                )

            except NotImplementedError as e:
                print(f"\n⚠️  {e}\n")
                sys.exit(1)
            except Exception as e:
                result = EvalResult(
                    case_id=case.id,
                    category=case.category,
                    tags=case.tags,
                    model_output="",
                    scores={},
                    passed=False,
                    latency_ms=0.0,
                    error=str(e)
                )
                case_passed = False

            results.append(result)
            f.write(json.dumps(result.__dict__) + "\n")

            # Live output
            status = "✓ PASS" if result.passed else "✗ FAIL"
            judge = result.scores.get("llm_judge_score", "—")
            judge_str = f"{judge:.2f}" if isinstance(judge, float) else str(judge)
            print(f"  [{result.case_id}] {status}  |  judge={judge_str}  |  {result.category}")

            if result.passed:
                passed += 1
            else:
                failed += 1

    # Summary
    total = passed + failed
    pass_rate = (passed / total * 100) if total else 0

    print(f"\n{'─'*60}")
    print(f"  Results: {passed}/{total} passed  ({pass_rate:.0f}%)")
    print(f"  Output:  {output_path}")
    print(f"{'─'*60}\n")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="echo-eval: LLM eval runner")
    parser.add_argument("--category", help="Filter by category (returns, billing, shipping, product, refusal)")
    parser.add_argument("--tag",      help="Filter by tag (e.g. safety, tone, escalation)")
    parser.add_argument("--case",     help="Run a single case by ID (e.g. cs_012)")
    parser.add_argument("--output",   default="results/results.jsonl", help="Output path for JSONL results")
    args = parser.parse_args()

    cases = load_cases(category=args.category, tag=args.tag, case_id=args.case)
    run_evals(cases, output_path=args.output)
