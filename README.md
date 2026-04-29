# echo-eval

**A usecase-driven LLM eval pipeline for customer support chatbots.**

Built for the *Echoes in AI* newsletter — Issue 04: LLM Evals from Scratch.

This repo is the hands-on companion to the [Substack post](#) which covers
the theory: what evals are, the taxonomy, how to choose metrics, and why the
wrong metric gives you false confidence. Read that first if you're new to evals.

---

## The usecase

You've built a customer support chatbot that handles returns, billing, product
questions, and shipping inquiries. It's live. How do you know if it's getting
worse after every prompt change?

This repo gives you a repeatable eval pipeline with 15 realistic test cases
across 5 categories, three layers of scoring, and a CLI you can wire into CI/CD.

---

## Repo structure

```
echo-eval/
├── data/
│   └── test_cases.json       # 15 realistic support conversations
├── evals/
│   ├── exact_match.py        # keyword presence / absence (deterministic)
│   ├── llm_judge.py          # tone, empathy, quality (LLM-as-judge)
│   └── refusal_check.py      # safety and boundary checking
├── models.py                 # EvalCase and EvalResult dataclasses
├── runner.py                 # main eval loop + CLI
├── report.py                 # report generator
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/your-username/echo-eval
cd echo-eval
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

Open `runner.py` and replace the `call_chatbot()` stub with your chatbot:

```python
# Example: OpenAI chatbot
from openai import OpenAI
client = OpenAI()

SYSTEM_PROMPT = "You are a helpful customer support agent for an outdoor gear company..."

def call_chatbot(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content
```

Then run:

```bash
# Run all 15 cases
python runner.py

# Run only return-related cases
python runner.py --category returns

# Run only safety cases
python runner.py --tag safety

# Run a single emotionally sensitive case
python runner.py --case cs_012

# Generate the report
python report.py

# Show only failures with model output
python report.py --failures-only --show-output
```

---

## The three scoring layers

### Layer 1 — Deterministic (`evals/exact_match.py`)

Keyword checks. Fast, cheap, no API calls needed.

- **keyword_presence** — required phrases must appear (e.g. "30-day", "refund")
- **keyword_absence** — forbidden phrases must not appear (e.g. "call us", "impossible")

These are your first filter. Run them locally before every push.

### Layer 2 — LLM-as-Judge (`evals/llm_judge.py`)

A second model (`gpt-4o-mini`) scores the response against a natural language
rubric. Used for dimensions that can't be checked with keywords:

- Does this response de-escalate an angry customer?
- Is this empathetic without being intrusive?
- Does it give a clear next step or just apologise?

Returns a 0.0–1.0 score, a one-sentence reasoning, and a list of specific gaps.

### Layer 3 — Refusal check (`evals/refusal_check.py`)

Safety-specific scoring for cases tagged `refusal` or `off-topic`.

- Did the bot refuse when it should have?
- Did it stay in character when a prompt injection was attempted?
- Did it accidentally comply with a request it should have blocked?

---

## Pass thresholds

| Scorer             | Threshold | Why                                                      |
|--------------------|-----------|----------------------------------------------------------|
| keyword_presence   | 1.0       | Binary — the phrase is either there or it isn't          |
| keyword_absence    | 1.0       | Binary — forbidden phrase absence is non-negotiable      |
| llm_judge_score    | 0.7       | Allows partial quality; 0.7+ is a genuinely good response|
| refusal_detected   | 1.0       | Safety is binary — no partial credit for almost refusing |

---

## Test case categories

| Category | Cases | What's tested                                            |
|----------|-------|----------------------------------------------------------|
| returns  | 4     | Standard policy, edge cases (defect), emotional context  |
| billing  | 3     | Double charges, escalation, cancellation                 |
| product  | 3     | Product info, comparisons, availability                  |
| shipping | 3     | Lost orders, international, deadline pressure            |
| refusal  | 3     | Privacy request, policy bypass, prompt injection         |

The most important case to pay attention to: **cs_012** — a customer returning
a gift bought by a recently deceased spouse. This is your empathy stress test.
A bot that scores 0.9 on everything else but fails cs_012 is not ready for production.

---

## Adding your own test cases

Add entries to `data/test_cases.json`:

```json
{
  "id": "cs_016",
  "category": "billing",
  "tags": ["fraud", "escalation"],
  "user_message": "Someone used my card without my permission on your site.",
  "expected_contains": ["fraud", "secure", "investigate"],
  "criteria": "The response must take the fraud claim seriously, not minimise it, explain the investigation process, and provide a clear timeline.",
  "should_not_contain": ["your fault", "check your account"]
}
```

Every bug your users surface is a new test case. That's how your regression
suite grows into institutional memory.

---

## Wiring into CI/CD

Add a GitHub Actions workflow to run evals on every PR:

```yaml
# .github/workflows/evals.yml
name: Run Evals

on: [pull_request]

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python runner.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - run: python report.py --failures-only
```

This way every prompt change is automatically evaluated against your full
test suite before it merges.

---

## Companion post

→ [Echoes in AI — Issue 04: LLM Evals from Scratch](#)

The Substack post covers:
- What evals actually are (and what they aren't)
- The six eval types you need to know
- Why each metric is recommended — and why the alternatives fail you
- The full `echo-eval` framework explained step by step

---

*Echoes in AI — monthly deep dives on LLMs, SLMs, and prompt engineering.*
