"""
learnlens/probes/consistency.py

ConsistencyProbe: does the agent make the same decision when the same
observation is rephrased with different surface text?

Inconsistency = surface pattern matching, not semantic reasoning.

Method:
  1. Run episode to collect a mid-episode pivot observation.
  2. Rephrase with 5 templates.
  3. Call agent_fn on each rephrasing (NO extra env steps).
  4. Score = fraction of times agent picks the majority action.
"""

from __future__ import annotations
from typing import Callable
from learnlens.probes.base import BaseProbe, StepTrace

PARAPHRASE_TEMPLATES: list[str] = [
    "{obs}",
    "Current state: {obs}",
    "You observe the following:\n{obs}",
    "Env state -- {obs}",
    "Step {step} observation:\n{obs}",
]


class ConsistencyProbe(BaseProbe):
    """
    Measures decision stability across surface rephrasing.
    Does NOT re-run the environment for each paraphrase.
    """

    def evaluate(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
    ) -> float:
        agreement_rates: list[float] = []

        for i in range(n_episodes):
            trace = self._run_episode(agent_fn, seed=i)
            if not trace.steps:
                continue
            pivot = trace.steps[len(trace.steps) // 2]
            agreement_rates.append(self._measure_agreement(agent_fn, pivot))

        return float(self._safe_mean(agreement_rates))

    def _measure_agreement(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        pivot: StepTrace,
    ) -> float:
        actions: list[str] = []
        for template in PARAPHRASE_TEMPLATES:
            rephrased = template.format(obs=pivot.observation, step=pivot.step_num)
            output = agent_fn(rephrased)
            action = output[0] if isinstance(output, tuple) else output
            actions.append(action.strip())
        if not actions:
            return 0.5
        majority = max(set(actions), key=actions.count)
        return float(actions.count(majority) / len(PARAPHRASE_TEMPLATES))