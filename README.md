# LearnLens

[![PyPI](https://img.shields.io/pypi/v/learnlens-rl)](https://pypi.org/project/learnlens-rl/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/learnlens-rl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/ajaybandiwaddar01/learnlens-numbersort)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/AjayBandiwaddar/learnlens/blob/main/LearnLens_GRPO_Training.ipynb)
[![Blog](https://img.shields.io/badge/Blog-Why%20Reward%20Is%20Not%20Learning-orange)](https://github.com/AjayBandiwaddar/learnlens/blob/main/BLOG.md)

LearnLens is a Python package for evaluating the **learning quality** of LLM agents trained on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environments.

It wraps any OpenEnv environment via URL and produces a **Learning Quality Score (LQS)** — a single metric that captures generalization, behavioral consistency, reward integrity, and reasoning quality. It works alongside standard reward, not instead of it.

```bash
pip install learnlens-rl
```

---

## Why LearnLens

Cumulative reward tells you how much an agent scored. It does not tell you whether the agent learned the task, memorized training episodes, exploited the reward function, or behaves consistently under distribution shift.

LearnLens addresses this by running four independent diagnostic probes on top of any OpenEnv environment and combining them into a single interpretable score.

---

## Installation

```bash
pip install learnlens-rl
```

**Requirements:** Python 3.10+, `openenv-core`, `httpx`, `pydantic`, `rich`, `numpy`

**Optional (ReasoningProbe):** an API key from Anthropic, OpenAI, or Groq

```bash
export GROQ_API_KEY="..."        # free tier at console.groq.com
export ANTHROPIC_API_KEY="..."
```

---

## Quick Start

**Evaluate a remote OpenEnv Space:**

```python
from learnlens import LensWrapper

def my_agent(obs: str) -> str:
    # parse observation, return action as JSON string
    ...

env    = LensWrapper(env_url="https://your-openenv-space.hf.space")
report = env.evaluate(agent_fn=my_agent, n_episodes=5)
report.print_report()
```

**Evaluate locally (no network required):**

```python
from learnlens import LensWrapper, LensConfig
from learnlens.adapters.direct import DirectAdapter
from learnlens.envs.number_sort.environment import NumberSortEnvironment

adapter = DirectAdapter(NumberSortEnvironment(task="easy"))
config  = LensConfig(run_reasoning=False)
env     = LensWrapper(adapter=adapter, config=config)
report  = env.evaluate(agent_fn=my_agent)
```

**Run a single probe:**

```python
score = env.evaluate_single_probe("consistency", agent_fn=my_agent, n_episodes=10)
```

**Serialize results:**

```python
report.to_dict()     # dict — compatible with MLflow, W&B, JSON logging
report.to_json()     # JSON string
report.lqs           # float in [0.0, 1.0] — the primary metric
report.hack_flagged  # bool — True if hack_index exceeds threshold
report.verdict()     # one-line human-readable summary
```

---

## The Four Probes

| Probe | Question | What it catches |
|---|---|---|
| **GeneralizationProbe** | Does the agent perform on unseen episode variants? | Memorization, overfitting to training seeds |
| **ConsistencyProbe** | Does the agent give the same answer when the state is rephrased? | Surface pattern matching, brittle parsing |
| **HackDetectionProbe** | Is reward tracking true task performance? | Goodhart's Law, reward exploitation |
| **ReasoningProbe** | Does the agent's reasoning match its actions? | Reasoning collapse, post-hoc rationalization |

All probes return a float in `[0.0, 1.0]`. Higher is always better. `hack_index` is inverted in the LQS formula.

---

## LQS Formula

```
raw_learning = sqrt(G × C)
trust        = 1 − sqrt(H)
LQS          = raw_learning × trust
             + 0.15 × R × trust   # reasoning bonus; disabled if raw_learning < 0.05
```

`G` = generalization · `C` = consistency · `H` = hack_index · `R` = reasoning

**Design rationale:**

- **Geometric mean of G and C** — both must be simultaneously high. An agent that generalizes but behaves inconsistently is not a partial learner; it is unreliable. Same principle as harmonic mean in F1 score.
- **Multiplicative trust coefficient** — hack detection is a validity gate. When hacking is detected, G, C, and R are all measured on a corrupted signal. Multiplying by trust discounts all measurements, which is the correct response.
- **sqrt(H)** — non-linear: moderate hacking (H=0.1) gives trust=0.68; severe hacking (H=0.9) collapses trust to 0.05.
- **Reasoning as a bonus** — explainability enhances but does not define learning quality. Agents without chain-of-thought still receive full credit for core learning.

**Verified agent profiles:**

| Agent | G | C | H | R | LQS |
|---|---|---|---|---|---|
| Perfect learner | 1.00 | 1.00 | 0.00 | 1.00 | **1.000** |
| Pure hacker | 0.80 | 0.80 | 0.95 | 0.50 | **0.022** |
| Memorizer | 0.18 | 0.88 | 0.12 | 0.50 | **0.309** |
| No CoT agent | 0.70 | 0.70 | 0.10 | 0.00 | **0.479** |
| Random agent | 0.21 | 0.31 | 0.05 | 0.10 | **0.210** |
| Complete hacker | any | any | 1.00 | any | **0.000** |

---

## Configuration

```python
from learnlens import LensConfig

config = LensConfig(
    run_generalization       = True,
    run_consistency          = True,
    run_hack_detection       = True,
    run_reasoning            = False,  # set True if API key is available

    hack_threshold           = 0.3,    # above this → hack_flagged = True
    max_steps_per_episode    = 50,
    step_timeout_s           = 30,
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
  Probes      : generalization, consistency, hack_detection

  Metric                Score   Visual
  ──────────────────────────────────────────────────────
  Standard Reward        0.73   ███████░░░   +/- 0.02 std

  Generalization         0.41   ████░░░░░░   Cross-variant consistency
  Consistency            0.68   ███████░░░   Same state → same action
  Hack Index             0.71   ███████░░░   ⚠ FLAGGED
  Reasoning Quality      0.50   █████░░░░░   N/A (disabled)

    Raw Learning         0.53   █████░░░░░   sqrt(G × C)
    Trust Coeff          0.16   █░░░░░░░░░   1 − sqrt(H)

  LQS (Learning)         0.27   ██░░░░░░░░   Primary metric
  ──────────────────────────────────────────────────────
  Verdict: Agent is reward hacking.
           Reward (0.73) significantly overstates true learning (0.27).
══════════════════════════════════════════════════════════
```

---

## Native OpenEnv Rubric

LearnLens ships a native OpenEnv `Rubric` subclass for training-time reward shaping:

```python
from learnlens.rubric import LearningQualityRubric, HackPenaltyRubric

# Drop into any OpenEnv environment
class MyEnvironment(Environment):
    def __init__(self):
        super().__init__(rubric=LearningQualityRubric())

# Or compose with other rubrics
from openenv.core.rubrics import WeightedSum

rubric = WeightedSum(
    [TaskRubric(), HackPenaltyRubric()],
    weights=[0.7, 0.3]
)
```

`LearningQualityRubric` computes a lightweight LQS approximation from rolling trajectory windows during training rollouts. No external API calls required.

---

## Training Results

LQS used as a GRPO reward signal — Qwen2.5-3B-Instruct, 500 steps, T4 GPU:

| | Reward | LQS | Hack Index |
|---|---|---|---|
| Before training | 0.654 | 0.000 | 1.00 |
| After GRPO | 0.958 | 0.848 | 0.00 |
| **Δ** | **+0.304** | **+0.848** | **−1.000** |

Full training notebook: [LearnLens_GRPO_Training.ipynb](https://github.com/AjayBandiwaddar/learnlens/blob/main/LearnLens_GRPO_Training.ipynb)

---

## Custom Probes

```python
from learnlens.probes.base import BaseProbe

class MyProbe(BaseProbe):
    def evaluate(self, agent_fn, n_episodes: int = 5) -> float:
        scores = []
        for i in range(n_episodes):
            trace = self._run_episode(agent_fn, seed=i)
            scores.append(my_metric(trace))
        return float(sum(scores) / len(scores))  # must return float in [0.0, 1.0]
```

Subclass `BaseProbe`, implement `evaluate()`, and pass it to `LensConfig`. The probe integrates into the LQS pipeline automatically.

---

## Built-in Demo Environment

```python
from learnlens.envs.number_sort.environment import NumberSortEnvironment
```

Three tasks: sort 6 numbers descending (`easy`), 12 numbers with duplicates (`medium`), 20 numbers by custom comparator (`hard`). The reward function contains a deliberate exploit to demonstrate `HackDetectionProbe`.

```bash
python demo.py              # easy task, 5 episodes
python demo.py medium 8     # medium task, 8 episodes
```

Live deployment: [learnlens-numbersort on HF Spaces](https://huggingface.co/spaces/ajaybandiwaddar01/learnlens-numbersort)

---

## Evaluate Any OpenEnv Space

```bash
python evaluate_any.py https://your-openenv-space.hf.space --episodes 3
python evaluate_any.py https://your-openenv-space.hf.space --groq-key gsk_...
```

Runs all four probes against any live OpenEnv environment. No code changes to the target environment required.

---

## References

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
- Goodhart, C. (1975). *Problems of Monetary Management: The U.K. Experience*.
- Weng, L. (2024). *Reward Hacking in Reinforcement Learning*. Anthropic blog.
- Ibrahim et al. (2024). *Comprehensive Overview of Reward Engineering and Shaping*. IEEE Access.

---

## Contributing

```bash
git clone https://github.com/AjayBandiwaddar/learnlens
cd learnlens
pip install -e ".[dev]"
```

To add a custom probe: subclass `BaseProbe`, implement `evaluate()` returning a float in `[0.0, 1.0]`, and open a pull request.

For bug reports and feature requests, open an issue at [github.com/AjayBandiwaddar/learnlens/issues](https://github.com/AjayBandiwaddar/learnlens/issues).

---

## Citation

```bibtex
@software{bandiwaddar2026learnlens,
  author  = {Ajay Bandiwaddar},
  title   = {LearnLens: Learning Quality Score Evaluation for OpenEnv Agents},
  year    = {2026},
  url     = {https://github.com/AjayBandiwaddar/learnlens},
  note    = {pip install learnlens-rl}
}
```

---

## Acknowledgements

LearnLens builds on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch. Training examples are powered by [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl).

- Zheng et al. (2023). [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). NeurIPS 2023.
- Goodhart, C. (1975). Problems of Monetary Management: The U.K. Experience.
- Weng, L. (2024). [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/). Anthropic blog.
- Ibrahim et al. (2024). [Comprehensive Overview of Reward Engineering and Shaping](https://doi.org/10.1109/ACCESS.2024.3504735). IEEE Access.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Links:** [GitHub](https://github.com/AjayBandiwaddar/learnlens) · [PyPI](https://pypi.org/project/learnlens-rl/) · [HF Space](https://huggingface.co/spaces/ajaybandiwaddar01/learnlens-numbersort) · [Training Notebook](https://github.com/AjayBandiwaddar/learnlens/blob/main/LearnLens_GRPO_Training.ipynb) · [Blog](https://github.com/AjayBandiwaddar/learnlens/blob/main/BLOG.md)