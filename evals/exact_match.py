"""
evals/exact_match.py
--------------------
Deterministic scorers — fast, cheap, no LLM calls needed.

These check structural and policy compliance:
  - keyword_presence : required phrases must appear in the output
  - keyword_absence  : forbidden phrases must NOT appear in the output

Both return 1.0 (pass) or 0.0 (fail) and record which strings caused failure.

When to use
-----------
Use these as your first layer of defense. They run in microseconds and
catch the most obvious failures — a return policy response that never
mentions "30-day", or a refusal response that accidentally includes the
thing it was supposed to refuse.
"""

from models import EvalCase


def keyword_presence_scorer(case: EvalCase, output: str) -> dict:
    """
    Check that all strings in case.expected_contains appear in the output.
    Case-insensitive. Returns 1.0 if all present, 0.0 if any are missing.
    Also returns a list of missing terms for debugging.
    """
    if not case.expected_contains:
        return {}

    output_lower = output.lower()
    missing = [
        term for term in case.expected_contains
        if term.lower() not in output_lower
    ]

    return {
        "keyword_presence": 0.0 if missing else 1.0,
        "keyword_presence_missing": missing  # surfaced in JSONL report
    }


def keyword_absence_scorer(case: EvalCase, output: str) -> dict:
    """
    Check that no strings in case.should_not_contain appear in the output.
    Case-insensitive. Returns 1.0 if all absent, 0.0 if any are found.
    Also returns a list of found violations for debugging.
    """
    if not case.should_not_contain:
        return {}

    output_lower = output.lower()
    found = [
        term for term in case.should_not_contain
        if term.lower() in output_lower
    ]

    return {
        "keyword_absence": 0.0 if found else 1.0,
        "keyword_absence_violations": found  # surfaced in JSONL report
    }
