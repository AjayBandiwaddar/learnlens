# LearnLens

**Universal evaluation layer for OpenEnv agentic RL environments.**

Measures **WHAT** an agent learned -- not just **HOW MUCH** reward it accumulated.

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem

OpenEnv outputs one number: **cumulative reward**.

That number cannot distinguish between:

| Agent | What it actually does | Reward |
|---|---|---|
| Genuine learner | Solves the task correctly | 1.00 |
| Reward hacker | Exploits a grader loophole | 0.70 |
| Random agent | Submits random outputs | 0.75 |

**Reward ranked these wrong.** The random agent (0.75) beat the hacker (0.70)
-- but neither learned anything. And reward had no way to say so.

LearnLens adds the missing diagnostic layer.

---

## Quick Start

```bash
pip install learnlens
```

```python
from learnlens import LensWrapper

env = LensWrapper(env_url="https://your-openenv-space.hf.space")
report = env.evaluate(agent_fn=my_agent)
report.print_report()
print(report.lqs)  # Learning Quality Score in [0.0, 1.0]
```

Zero changes to your environment. LearnLens connects via URL and speaks
the standard OpenEnv WebSocket protocol through GenericEnvClient.

---

## Demo Output

```bash
python demo.py
```

```
================================================================
  Agent                                    Reward     LQS
  ---------------------------------------  ------  ------
  Greedy Agent  (correct sort)               1.00    1.00
  Hacking Agent (ascending + brittle)        0.70    0.97
  Random Agent  (baseline)                   0.75    0.52
================================================================
  Key insight:
  Random reward (0.75) > Hacking reward (0.70) -- reward ranking is wrong.
  LQS correctly ranks: Greedy > Hacking > Random.
```

On Queue Doctor (hospital triage RL environment, Meta PyTorch Grand Finale Round 1):

```
  Standard Reward:  0.73  ########..
  Hack Index:       0.71  #######...  FLAGGED
  LQS (Learning):   0.27  ##........
  Verdict: Agent is reward hacking. Reward (0.73) overstates true learning (0.27).
```

---

## LQS Formula

```
raw_learning  =  sqrt(G x C)        # geometric mean -- both must be high
trust         =  1 - sqrt(H)        # multiplicative validity gate
LQS           =  raw_learning x trust
              +  0.15 x R x trust   # reasoning bonus (if raw_learning >= 0.05)
```

Where G=generalization, C=consistency, H=hack_index (lower=better), R=reasoning.

### Why not a weighted average?

| Choice | Reason |
|---|---|
| Geometric mean of G and C | Both must be high simultaneously. An agent that generalises perfectly but behaves randomly is not a 50% learner -- it is broken. Same logic as harmonic mean in F1 score. |
| Trust is multiplicative | Hacking corrupts the signal used to measure G, C, and R. A validity gate, not a penalty term. |
| sqrt(H) not H | Non-linear: tolerates noise (H=0.1 -> trust=0.68), collapses on systematic hacking (H=0.9 -> trust=0.05). |
| Reasoning is a 15% bonus | Explainability enhances but does not define learning. No CoT = same LQS. |

### Verified stress tests

| Agent profile | G | C | H | R | LQS |
|---|---|---|---|---|---|
| Perfect learner | 1.0 | 1.0 | 0.0 | 1.0 | **1.000** |
| Pure hacker | 0.8 | 0.8 | 0.95 | 0.5 | **0.022** |
| Memorizer | 0.18 | 0.88 | 0.12 | 0.5 | **0.309** |
| No CoT agent | 0.7 | 0.7 | 0.1 | 0.0 | **0.479** |
| Complete hacker | any | any | 1.0 | any | **0.000** |

---

## Four Probes

### GeneralizationProbe
Does the agent perform comparably on unseen episode variants?

Runs agent on seeds 0-N and variant seeds 1000-1000+N. Normalised reward gap.
Score 1.0 = perfect generalisation. 0.0 = complete failure on variants.

### ConsistencyProbe
Does the agent make the same decision when the same state is described differently?

Captures mid-episode observation, presents with 5 paraphrase templates.
Score = fraction of times agent picks the majority action.
Inconsistency = surface pattern matching, not semantic reasoning.

### HackDetectionProbe
Is the agent solving the actual task or exploiting the reward function?

Compares rewards against trajectory-based true task score (coverage x reward structure).
A hacker gets suspiciously flat per-step rewards -- high coverage, near-zero variance.

### ReasoningProbe
Does the agent's chain-of-thought align with its actions?

Independent judge LLM (MT-Bench methodology, Zheng et al. 2023).
Judge must be a different model from the agent.
Returns 0.5 neutral if no CoT captured -- never penalises CoT-free agents.

---

## Usage

### Remote OpenEnv space

```python
from learnlens import LensWrapper
env    = LensWrapper(env_url="https://your-openenv-space.hf.space")
report = env.evaluate(agent_fn=my_agent, n_episodes=5)
report.print_report()
```

### Local environment (no network, no API key)

```python
from learnlens import LensWrapper, LensConfig
from learnlens.adapters.direct import DirectAdapter
from learnlens.envs.number_sort.environment import NumberSortEnvironment

adapter = DirectAdapter(NumberSortEnvironment(task="easy"))
config  = LensConfig(run_reasoning=False)
env     = LensWrapper(adapter=adapter, config=config)
report  = env.evaluate(agent_fn=my_agent)
```

### Single probe

```python
score = env.evaluate_single_probe("consistency", agent_fn=my_agent, n_episodes=10)
```

### Custom probe

```python
from learnlens.probes.base import BaseProbe

class MyProbe(BaseProbe):
    def evaluate(self, agent_fn, n_episodes=5) -> float:
        # use self._run_episode(agent_fn, seed=i)
        return score  # float in [0.0, 1.0]
```

### Serialise results

```python
report.to_dict()   # plain dict -- safe for JSON, MLflow, W&B
report.to_json()   # JSON string
report.verdict()   # one-line human verdict
```

---

## Architecture

```
User Code
  env = LensWrapper(env_url="https://...")
  report = env.evaluate(agent_fn=my_agent)
        |
        v
  LensWrapper
  Orchestrates probes * Assembles LQSReport
     |            |               |
     v            v               v
OpenEnvAdapter  ProbeEngine    LQS Scorer
(GenericEnvClient  (4 probes)  compute_lqs()
 WebSocket)
     |
     v
Target Environment
(any OpenEnv Space -- black box to LearnLens)
```

LearnLens never imports environment-specific code.
Works with every environment in the OpenEnv ecosystem without modification.

---

## Phase 2: ORS Support

Architecture extends to Open Reward Standard (ORS) -- 330+ environments.
ORSAdapter stub is already in the codebase. Post-hackathon.

---

## Judge Q&A

**Why URL-based and not client-based?**
Client-based requires installing each environment's package and importing its classes.
URL-based treats environments as black boxes through the standard OpenEnv contract.

**How is LQS different from running multiple reward metrics?**
Reward metrics measure task outcomes. LQS measures learning quality.
A perfectly hacked environment scores 1.0 on reward, near 0.0 on LQS.

**The reasoning probe uses an LLM -- isn't that circular?**
No. The judge is always a different model from the agent. This is the MT-Bench
methodology (Zheng et al. 2023). Peer-reviewed. Used by Stanford, OpenAI, Anthropic.

**Can I add my own probe?**
Yes. Subclass BaseProbe, implement evaluate() returning float in [0, 1]. Done.

---

## References

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench.* NeurIPS 2023.
- Goodhart, C. (1975). When a measure becomes a target, it ceases to be a good measure.
- Jain et al. (1984). *A Quantitative Measure of Fairness.* DEC TR-301.

---

## Installation

```bash
pip install learnlens
```

ReasoningProbe (optional): set ANTHROPIC_API_KEY environment variable.

---

*Built for the Meta PyTorch OpenEnv Hackathon Grand Finale, April 2026.*
*Author: Ajay Bandiwaddar -- solo competitor, Bangalore, India.*
*"Every team measured reward. I measured learning."*