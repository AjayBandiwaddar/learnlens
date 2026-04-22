# LearnLens

> **Universal evaluation layer for OpenEnv agentic RL environments.**  
> Measures *what* an agent learned — not just *how much* reward it accumulated.

[![PyPI](https://img.shields.io/pypi/v/learnlens?color=blue)](https://pypi.org/project/learnlens/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HF Space](https://img.shields.io/badge/demo-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/ajaybandiwaddar01/learnlens-numbersort)

---

## Overview

OpenEnv outputs one number: **cumulative reward**. That number cannot distinguish between an agent that genuinely learned a skill, one that exploited a grader loophole, one that memorised episode patterns, or one that behaves inconsistently across semantically identical states.

LearnLens adds the missing diagnostic layer. It wraps **any** OpenEnv environment via URL — zero modifications to the target environment — and produces a **Learning Quality Score (LQS)** alongside four interpretable probe scores.

```bash
pip install learnlens
```

```python
from learnlens import LensWrapper

env    = LensWrapper(env_url="https://your-openenv-space.hf.space")
report = env.evaluate(agent_fn=my_agent)
report.print_report()
```

---

## The Problem LearnLens Solves

| Agent | Behaviour | Reward |
|---|---|---|
| Genuine learner | Solves the task correctly | 1.00 |
| Random agent | Submits random outputs | 0.75 |
| Reward hacker | Exploits a grader loophole | 0.70 |

Reward ranked these wrong. The random agent (0.75) outscored the hacker (0.70) — but neither learned anything meaningful. Reward had no way to say so.

**LQS correctly ranks them:**

| Agent | Reward | LQS |
|---|---|---|
| Genuine learner | 1.00 | **1.00** |
| Reward hacker | 0.70 | **0.97** |
| Random agent | 0.75 | **0.52** |

The hacker is at least consistent — it always applies the same exploit. The random agent is neither consistent nor generalising. LQS captures the difference. Reward cannot.

---

## Installation

```bash
pip install learnlens
```

**Requirements:** Python 3.10+, openenv-core, httpx, pydantic, rich, numpy.

**ReasoningProbe (optional):** Requires an API key for the LLM judge.  
Supported providers: Anthropic, OpenAI, Groq (free tier available).

```bash
export ANTHROPIC_API_KEY="..."   # or
export GROQ_API_KEY="..."        # free at console.groq.com
```

---

## Quick Start

### Evaluate a remote OpenEnv Space

```python
from learnlens import LensWrapper, LensConfig

def my_agent(observation: str) -> str:
    # Parse observation, return action as JSON string
    ...

env    = LensWrapper(env_url="https://your-space.hf.space")
report = env.evaluate(agent_fn=my_agent, n_episodes=5)
report.print_report()
```

### Evaluate locally (no network required)

```python
from learnlens import LensWrapper, LensConfig
from learnlens.adapters.direct import DirectAdapter
from learnlens.envs.number_sort.environment import NumberSortEnvironment

adapter = DirectAdapter(NumberSortEnvironment(task="easy"))
config  = LensConfig(run_reasoning=False)
env     = LensWrapper(adapter=adapter, config=config)
report  = env.evaluate(agent_fn=my_agent)
```

### Run a single probe

```python
score = env.evaluate_single_probe("consistency", agent_fn=my_agent, n_episodes=10)
```

### Serialise results for logging

```python
report.to_dict()   # dict — compatible with MLflow, W&B, JSON
report.to_json()   # JSON string
report.verdict()   # one-line human-readable verdict
```

### Enable reasoning evaluation with Groq (free)

```python
env = LensWrapper(
    env_url="https://your-space.hf.space",
    judge_model="llama-3.1-8b-instant",   # Groq free tier
    judge_api_key="gsk_...",
    config=LensConfig(run_reasoning=True)
)
```

---

## Sample Report

```
══════════════════════════════════════════════════════════
  LearnLens Evaluation Report
══════════════════════════════════════════════════════════
  Environment : https://your-space.hf.space
  Episodes    : 5
  Probes      : generalization, consistency, hack_detection, reasoning

  Metric                Score   Visual
  ──────────────────────────────────────────────────────
  Standard Reward        0.73   ███████░░░   +/- 0.02 std

  Generalization         0.41   ████░░░░░░   Cross-variant consistency
  Consistency            0.68   ███████░░░   Same state -> same action
  Hack Index             0.71   ███████░░░   ⚠ FLAGGED
  Reasoning Quality      0.55   █████░░░░░   CoT quality

    Raw Learning         0.53   █████░░░░░   sqrt(G x C)
    Trust Coeff          0.16   █░░░░░░░░░   1 - sqrt(H)

  LQS (Learning)         0.27   ██░░░░░░░░   Primary metric
  ──────────────────────────────────────────────────────
  Verdict: Agent is reward hacking.
           Reward (0.73) significantly overstates true learning (0.27).
══════════════════════════════════════════════════════════
```

---

## LQS Formula

```
raw_learning  =  sqrt(G × C)              # geometric mean of generalization and consistency
trust         =  1 − sqrt(H)              # multiplicative validity gate on hack index
LQS           =  raw_learning × trust
              +  0.15 × R × trust         # reasoning bonus (disabled if raw_learning < 0.05)
```

Where **G** = generalization, **C** = consistency, **H** = hack_index, **R** = reasoning.

### Design decisions

| Decision | Rationale |
|---|---|
| Geometric mean of G and C | Both must be simultaneously high. An agent that generalises perfectly but behaves randomly is not a 50% learner — it is broken. Same principle as harmonic mean in F1 score. |
| Trust is multiplicative, not additive | Hacking corrupts the signal used to measure G, C, and R. When hacking is detected, no other measurement can be trusted. A validity gate discounts the entire stack — not just subtracts a penalty. |
| sqrt(H) not H | Non-linear: moderate hacking (H=0.1) gives trust=0.68 (tolerated), severe hacking (H=0.9) gives trust=0.05 (collapsed). |
| Reasoning is a 15% bonus | Explainability enhances but does not define learning. An agent with no chain-of-thought still achieves full LQS credit for its core learning. |
| Reasoning gated on raw_learning ≥ 0.05 | Reasoning quality is irrelevant if core learning has completely failed. |

### Verified agent profiles

| Agent | G | C | H | R | LQS |
|---|---|---|---|---|---|
| Perfect learner | 1.00 | 1.00 | 0.00 | 1.00 | **1.000** |
| Pure hacker | 0.80 | 0.80 | 0.95 | 0.50 | **0.022** |
| Memorizer | 0.18 | 0.88 | 0.12 | 0.50 | **0.309** |
| No CoT agent | 0.70 | 0.70 | 0.10 | 0.00 | **0.479** |
| Random agent | 0.21 | 0.31 | 0.05 | 0.10 | **0.210** |
| Complete hacker | any | any | 1.00 | any | **0.000** |

---

## The Four Probes

### GeneralizationProbe
**Does the agent perform comparably on unseen episode variants?**

Runs the agent on base seeds (0–N) and variant seeds (1000–1000+N). The score measures the normalised reward gap between base and variant performance. A score of 1.0 indicates perfect transfer; 0.0 indicates complete failure on variants — the agent memorised, not learned.

### ConsistencyProbe
**Does the agent make the same decision when the same state is described differently?**

Captures a mid-episode observation and presents it with five paraphrase templates — same semantic content, different surface format. The agent is called five times without advancing environment state. Score = fraction of times the agent picks the majority action. Brittle agents that only parse raw JSON fail on four of five templates.

### HackDetectionProbe
**Is the agent solving the task or exploiting the reward function?**

Computes an environment-agnostic true task score from trajectory analysis — specifically, reward structure and coverage across steps. A hacking agent produces unnaturally uniform per-step rewards (same exploit applied every step). The hack_index measures the normalised gap between reward and true task performance. This probe is most powerful on multi-step MDP environments.

### ReasoningProbe
**Does the agent's stated reasoning align with its actions?**

An independent judge LLM scores agent chain-of-thought on three dimensions: relevance (did the agent reference key state variables?), coherence (does the reasoning logically support the action?), and uncertainty (did the agent appropriately flag ambiguity?). The judge is always a **different model** from the agent — MT-Bench methodology (Zheng et al., NeurIPS 2023). Returns 0.5 neutral if no chain-of-thought is captured or no API key is configured. Never penalises CoT-free agents.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Code                          │
│  env = LensWrapper(env_url="https://...")               │
│  report = env.evaluate(agent_fn=my_agent)               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    LensWrapper                          │
│  Orchestrates probes · Assembles LQSReport              │
└────────┬──────────────────┬──────────────────┬──────────┘
         │                  │                  │
         ▼                  ▼                  ▼
  OpenEnvAdapter       ProbeEngine         LQS Scorer
  GenericEnvClient     4 probes            compute_lqs()
  WebSocket protocol
         │
         ▼
  Target Environment
  (any OpenEnv Space — black box to LearnLens)
  POST /reset · POST /step · GET /state · GET /health
```

LearnLens **never imports environment-specific code**. It communicates exclusively through the standard OpenEnv WebSocket protocol. Every environment in the OpenEnv ecosystem works without modification.

---

## Custom Probes

LearnLens is explicitly designed for extension.

```python
from learnlens.probes.base import BaseProbe

class MyProbe(BaseProbe):
    def evaluate(self, agent_fn, n_episodes: int = 5) -> float:
        scores = []
        for i in range(n_episodes):
            trace = self._run_episode(agent_fn, seed=i)
            # analyse trace.steps, trace.total_reward
            scores.append(my_metric(trace))
        return float(sum(scores) / len(scores))  # must return float in [0.0, 1.0]
```

Pass it to `LensConfig` and it integrates into the LQS pipeline automatically.

---

## Built-in Demo Environment: NumberSort

LearnLens ships with a complete OpenEnv-compatible environment for local demonstration.

```python
from learnlens.envs.number_sort.environment import NumberSortEnvironment
```

Three tasks: sort 6 numbers descending (easy), 12 numbers with duplicates (medium), 20 numbers by custom comparator (hard). The reward function contains a deliberate exploit — returning any permutation scores ≥ 0.70 — making reward hacking obvious and demonstrable.

Run the full demo:

```bash
python demo.py           # easy task, 5 episodes
python demo.py medium 8  # medium task, 8 episodes
```

A live deployment of NumberSort is available at:  
**https://huggingface.co/spaces/ajaybandiwaddar01/learnlens-numbersort**

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | OpenEnv adapter, 4 probes, NumberSort environment, PyPI |
| Phase 2 | 🔄 Planned | ORSAdapter — 330+ environments at openrewardstandard.io |
| Phase 3 | 🔄 Planned | Training loop integration, MLflow callback, LQS-as-reward-signal |

The ORSAdapter stub is already in the codebase (`learnlens/adapters/ors.py`). Phase 2 implementation maps ORS `/start` and MCP tool-calling protocol to the same probe interface with zero changes to Phase 1 code.

---

## References

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
- Goodhart, C. (1975). *Problems of Monetary Management.* (origin of Goodhart's Law)
- Jain, Chiu, Hawe (1984). *A Quantitative Measure of Fairness and Discrimination.* DEC TR-301.
- OpenEnv RFC #468 — Standardised agent evaluation metrics (gap addressed by LearnLens).

---

## Contributing

Issues and pull requests welcome at [github.com/AjayBandiwaddar/learnlens](https://github.com/AjayBandiwaddar/learnlens).

To add a probe, subclass `BaseProbe`, implement `evaluate()` returning a float in `[0.0, 1.0]`, and open a PR.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built for the Meta PyTorch OpenEnv Hackathon Grand Finale, April 2026.*  
*Author: Ajay Bandiwaddar — solo competitor, Bangalore, India.*  
*"Every team measured reward. I measured learning."*