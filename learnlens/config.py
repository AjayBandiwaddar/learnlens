"""
learnlens/config.py

LensConfig: probe selection and LQS formula parameters.
This is NOT a reward function. It controls which probes run
and how the Learning Quality Score is assembled.
"""

from dataclasses import dataclass


@dataclass
class LensConfig:
    """
    Configuration for a LearnLens evaluation run.

    Probe flags let you disable individual probes when you lack
    an API key (run_reasoning=False) or want faster evaluation.

    The LQS formula is NOT a weighted average -- see scorer.py.
    Formula: sqrt(G*C) * (1 - sqrt(H)) + 0.15*R*(1 - sqrt(H))
    The weight fields below are kept for documentation only.
    """

    # Probe selection
    run_generalization: bool = True
    run_consistency: bool = True
    run_hack_detection: bool = True
    run_reasoning: bool = True   # Requires ANTHROPIC_API_KEY

    # Legacy weights (not used in primary LQS formula)
    weight_generalization: float = 0.30
    weight_consistency: float = 0.25
    weight_hack_detection: float = 0.25
    weight_reasoning: float = 0.20

    # Probe tuning
    n_variants: int = 3
    n_paraphrases: int = 5
    hack_threshold: float = 0.3
    core_learning_threshold: float = 0.05

    # Episode settings
    step_timeout_s: int = 30
    max_steps_per_episode: int = 50

    def validate(self) -> None:
        """Raises ValueError if config is internally inconsistent."""
        total = (
            self.weight_generalization
            + self.weight_consistency
            + self.weight_hack_detection
            + self.weight_reasoning
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"LensConfig weights must sum to 1.0, got {total:.4f}."
            )
        if not (0 <= self.hack_threshold <= 1):
            raise ValueError(
                f"hack_threshold must be in [0, 1], got {self.hack_threshold}"
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