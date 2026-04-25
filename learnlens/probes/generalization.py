"""
learnlens/probes/generalization.py

GeneralizationProbe — does the agent generalise across episode variants,
or did it memorise the base episodes?

Method:
  Run agent on base seeds (0..n-1) and variant seeds (1000..1000+n-1).
  Measure normalised performance gap between the two sets.

Formula:
  gap   = |mean(base_rewards) - mean(variant_rewards)| / max(both)
  score = 1.0 - gap

Score interpretation:
  1.0 = perfect generalisation (same performance on unseen variants)
  0.0 = complete failure on variants (pure memorisation)
  0.5 = neutral (returned when environment ignores seeds)

If the environment produces identical episodes regardless of seed
(seed is ignored), base and variant rewards will be equal → score=1.0.
This is optimistic but not wrong — the probe loses signal rather
than incorrectly penalising the environment.

References:
  Generalisation in RL: Cobbe et al. (2019) "Quantifying Generalisation
  in Reinforcement Learning", arXiv:1812.02341
"""

from __future__ import annotations
from typing import Callable

from learnlens.probes.base import BaseProbe

VARIANT_SEED_OFFSET = 1_000


class GeneralizationProbe(BaseProbe):
    """
    Measures performance gap between base seeds and variant seeds.

    Attribute last_base_rewards is populated after evaluate() and can
    be used by LensWrapper to avoid re-running episodes for reward stats.
    """

    def __init__(self, adapter, config) -> None:
        super().__init__(adapter, config)
        self.last_base_rewards: list[float] = []

    def evaluate(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
    ) -> float:
        base_rewards:    list[float] = []
        variant_rewards: list[float] = []

        for i in range(n_episodes):
            base_rewards.append(
                self._run_episode(agent_fn, seed=i).total_reward
            )
            variant_rewards.append(
                self._run_episode(agent_fn, seed=i + VARIANT_SEED_OFFSET).total_reward
            )

        self.last_base_rewards = base_rewards
        return self._score(base_rewards, variant_rewards)

    @staticmethod
    def _score(base: list[float], variant: list[float]) -> float:
        if not base or not variant:
            return 0.5
        avg_base    = sum(base)    / len(base)
        avg_variant = sum(variant) / len(variant)
        denom = max(avg_base, avg_variant)
        if denom <= 0.0:
            # Both zero — environment gives no reward signal at all
            return 0.5
        gap = abs(avg_base - avg_variant) / denom
        return float(max(0.0, min(1.0, 1.0 - gap)))