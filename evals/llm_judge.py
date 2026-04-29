"""
evals/llm_judge.py
------------------
LLM-as-judge scorer — uses a second model to evaluate output quality
on dimensions that can't be checked deterministically.

For a customer support chatbot, this covers:
  - Tone and empathy (especially for sensitive cases like cs_012)
  - Whether the response actually helps the customer
  - Whether it follows the spirit of the instruction, not just the letter

Design decisions
----------------
1. We use a cheap, fast judge model (gpt-4o-mini) — not the same model
   being evaluated. This avoids self-evaluation bias.

2. The judge scores on a 0.0–1.0 scale using a structured rubric derived
   from case.criteria. We ask it to return JSON, not free text, so we can
   parse the score reliably.

3. We include the user's original message in the judge prompt so it can
   assess contextual appropriateness, not just output quality in isolation.

4. Temperature=0 for reproducibility. You want the same case scored the
   same way each run unless you're intentionally exploring variance.

Calibration note
----------------
Before trusting LLM judge scores in CI, validate them against 20–30 human
labels on your own test cases. Judge models have known biases:
  - Verbosity bias: longer responses score higher
  - Politeness bias: overly apologetic responses score higher
  - Position bias: if you show two outputs, the first scores higher

Mitigate by: keeping rubric criteria specific, using binary sub-questions
("Does the response mention a timeframe? Yes/No"), and spot-checking
borderline scores (0.5–0.75) manually.
"""

import json
import os
from openai import OpenAI
from models import EvalCase

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of customer support chatbot responses.
You will be given:
- The customer's message
- The chatbot's response
- An evaluation rubric

Score the response on the rubric from 0.0 to 1.0, where:
  1.0 = Fully meets all rubric criteria
  0.7 = Meets most criteria with minor gaps
  0.5 = Partially meets criteria with notable gaps
  0.3 = Fails most criteria
  0.0 = Completely fails or causes harm

Return ONLY valid JSON in this exact format, no other text:
{
  "score": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining the score>",
  "gaps": ["<specific gap 1>", "<specific gap 2>"]
}"""


def llm_judge_scorer(case: EvalCase, output: str) -> dict:
    """
    Score the chatbot output against case.criteria using an LLM judge.
    Returns the score, reasoning, and any identified gaps.
    """
    if not case.criteria:
        return {}

    user_prompt = f"""Customer message:
{case.user_message}

Chatbot response:
{output}

Evaluation rubric:
{case.criteria}

Score this response."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",          # cheap judge — swap for gpt-4o on critical cases
            temperature=0,                # reproducible scoring
            max_tokens=300,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        return {
            "llm_judge_score":     float(parsed.get("score", 0.0)),
            "llm_judge_reasoning": parsed.get("reasoning", ""),
            "llm_judge_gaps":      parsed.get("gaps", [])
        }

    except json.JSONDecodeError:
        # Judge returned malformed JSON — score as 0 and surface the raw output
        return {
            "llm_judge_score":     0.0,
            "llm_judge_reasoning": "Judge returned unparseable response",
            "llm_judge_gaps":      [raw[:200] if 'raw' in dir() else "no response"]
        }
    except Exception as e:
        return {
            "llm_judge_score":     0.0,
            "llm_judge_reasoning": f"Scorer error: {str(e)}",
            "llm_judge_gaps":      []
        }
