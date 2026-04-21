"""
learnlens/probes/generalization.py

GeneralizationProbe: does the agent perform comparably on unseen episode
variants, or did it memorise the base episodes?

Formula:
    gap = |mean(base_rewards) - mean(variant_rewards)| / max(both)
    generalization_score = 1 - gap

Score: 1.0 = perfect generalisation. 0.0 = complete failure on variants.
"""

from __future__ import annotations
from typing import Callable
from learnlens.probes.base import BaseProbe, EpisodeTrace

VARIANT_SEED_OFFSET = 1_000


class GeneralizationProbe(BaseProbe):
    """
    Runs agent on base seeds (0..n-1) and variant seeds (1000..1000+n-1).
    Measures normalised performance gap between the two sets.
    """

    def evaluate(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
    ) -> float:
        base_rewards: list[float] = []
        variant_rewards: list[float] = []

        for i in range(n_episodes):
            base_rewards.append(self._run_episode(agent_fn, seed=i).total_reward)
            variant_rewards.append(self._run_episode(agent_fn, seed=i + VARIANT_SEED_OFFSET).total_reward)

        return self._score(base_rewards, variant_rewards)

    @staticmethod
    def _score(base_rewards: list[float], variant_rewards: list[float]) -> float:
        if not base_rewards or not variant_rewards:
            return 0.5
        avg_base = sum(base_rewards) / len(base_rewards)
        avg_variant = sum(variant_rewards) / len(variant_rewards)
        denom = max(avg_base, avg_variant)
        if denom <= 0.0:
            return 0.5
        gap = abs(avg_base - avg_variant) / denom
        return float(max(0.0, min(1.0, 1.0 - gap)))