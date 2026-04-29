"""
models.py
---------
Core data structures for echo-eval.
All eval cases and results flow through these dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalCase:
    """
    A single eval test case.

    Fields
    ------
    id              : Unique identifier (e.g. "cs_001")
    category        : Task category (e.g. "returns", "billing", "refusal")
    user_message    : The raw user input sent to the chatbot
    criteria        : Natural language rubric used by the LLM judge scorer
    tags            : Free-form labels for filtering (e.g. ["safety", "tone"])
    expected_contains : List of strings the output must contain (case-insensitive)
    should_not_contain: List of strings the output must NOT contain (case-insensitive)
    """
    id: str
    category: str
    user_message: str
    criteria: str
    tags: list = field(default_factory=list)
    expected_contains: list = field(default_factory=list)
    should_not_contain: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    The result of running a single EvalCase through the eval pipeline.

    Fields
    ------
    case_id         : Links back to EvalCase.id
    category        : Copied from EvalCase for easy filtering
    tags            : Copied from EvalCase
    model_output    : Raw text output from the chatbot under evaluation
    scores          : Dict of scorer_name -> float (0.0–1.0)
    passed          : True if all scores >= their thresholds
    latency_ms      : Time taken to get the model response
    error           : Set if the model call or scoring raised an exception
    """
    case_id: str
    category: str
    tags: list
    model_output: str
    scores: dict
    passed: bool
    latency_ms: float
    error: Optional[str] = None
