"""
learnlens/config.py

LensConfig: probe selection and evaluation parameters.

The LQS formula is NOT a weighted average.
Formula: sqrt(G*C) * (1 - sqrt(H)) + 0.15*R*(1-sqrt(H))
Weight fields are kept for documentation and legacy compatibility only.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LensConfig:
    """
    Configuration for a LearnLens evaluation run.

    Disable probes you don't need:
        config = LensConfig(run_reasoning=False)   # no API key needed
        config = LensConfig(run_hack_detection=False)  # faster eval

    Disabling probes does NOT break validate(). Disabled probes
    contribute their neutral default to the LQS formula.
    """

    # ── Probe selection ────────────────────────────────────────────────
    run_generalization:  bool = True
    run_consistency:     bool = True
    run_hack_detection:  bool = True
    run_reasoning:       bool = True   # Requires API key

    # ── Legacy weight fields (NOT used in LQS formula) ─────────────────
    weight_generalization:  float = 0.30
    weight_consistency:     float = 0.25
    weight_hack_detection:  float = 0.25
    weight_reasoning:       float = 0.20

    # ── Probe tuning ───────────────────────────────────────────────────
    n_variants:              int   = 3
    n_paraphrases:           int   = 5
    hack_threshold:          float = 0.3
    core_learning_threshold: float = 0.05

    # ── Episode settings ───────────────────────────────────────────────
    step_timeout_s:        int = 30
    max_steps_per_episode: int = 50

    def validate(self) -> None:
        """
        Validates config. Does NOT check weight sums — weights are
        legacy fields unused in the formula. Raises ValueError on
        actual misconfigurations only.
        """
        if not (0.0 <= self.hack_threshold <= 1.0):
            raise ValueError(
                f"hack_threshold must be in [0, 1], got {self.hack_threshold}"
            )
        if not (0.0 <= self.core_learning_threshold <= 1.0):
            raise ValueError(
                f"core_learning_threshold must be in [0, 1], "
                f"got {self.core_learning_threshold}"
            )
        if self.max_steps_per_episode < 1:
            raise ValueError(
                f"max_steps_per_episode must be >= 1, "
                f"got {self.max_steps_per_episode}"
            )
        if self.n_paraphrases < 1:
            raise ValueError(
                f"n_paraphrases must be >= 1, got {self.n_paraphrases}"
            )

    def active_probes(self) -> list[str]:
        """Return names of probes that will run."""
        probes = []
        if self.run_generalization:
            probes.append("generalization")
        if self.run_consistency:
            probes.append("consistency")
        if self.run_hack_detection:
            probes.append("hack_detection")
        if self.run_reasoning:
            probes.append("reasoning")
        return probes

    def any_probe_active(self) -> bool:
        return any([
            self.run_generalization,
            self.run_consistency,
            self.run_hack_detection,
            self.run_reasoning,
        ])