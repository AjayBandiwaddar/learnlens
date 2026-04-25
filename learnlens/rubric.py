"""
learnlens/rubric.py

LearningQualityRubric -- LQS as a native OpenEnv Rubric.

This makes LearnLens a first-class citizen inside the OpenEnv ecosystem.
Any environment can use LQS as its training reward signal in one line:

    from learnlens.rubric import LearningQualityRubric

    class MyEnvironment(Environment):
        def __init__(self):
            super().__init__(rubric=LearningQualityRubric())
            ...

Or as a composable component alongside other rubrics:

    from openenv.core.rubrics import WeightedSum
    from learnlens.rubric import LearningQualityRubric, HackPenaltyRubric

    rubric = WeightedSum([
        MyTaskRubric(),
        HackPenaltyRubric(),
    ], weights=[0.7, 0.3])

Design:
  OpenEnv Rubric.forward(action, observation) -> float
  LearningQualityRubric maintains a rolling window of recent
  (action, observation, reward) tuples per session and computes
  LQS from trajectory signals — no external calls needed.

  This is different from the full LensWrapper evaluation (which runs
  separate probe episodes). The Rubric variant computes a lightweight
  LQS approximation in real-time during training rollouts.

  Lightweight approximation (suitable for dense reward shaping):
    G_approx: rolling reward variance across recent episodes (stability)
    C_approx: action consistency under same-context calls (coherence)
    H_approx: reward flatness detection (hack signal)
    R_approx: not computed here (no judge model in training loop)

  Full evaluation: use LensWrapper.evaluate() post-training.
  Real-time shaping: use LearningQualityRubric in environment rubric.

Note: requires openenv-core. Gracefully stubs if not installed.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from learnlens.scorer import compute_lqs

# Maximum steps to keep in rolling window per session
_WINDOW = 20


class _RubricBase:
    """
    Minimal stub used when openenv-core is not installed.
    Allows learnlens to be imported without openenv-core.
    """
    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError

    def __call__(self, action: Any, observation: Any) -> float:
        return self.forward(action, observation)


def _get_rubric_base():
    """Return openenv Rubric base class if available, else our stub."""
    try:
        from openenv.core.rubrics import Rubric
        return Rubric
    except ImportError:
        return _RubricBase


class LearningQualityRubric(_get_rubric_base()):  # type: ignore[misc]
    """
    Native OpenEnv Rubric that computes an LQS-based reward signal.

    Drop into any OpenEnv environment as a training reward:

        class MyEnv(Environment):
            def __init__(self):
                super().__init__(rubric=LearningQualityRubric())

    Reward signal properties:
    - Penalises flat/uniform reward patterns (hack detection)
    - Rewards consistent action selection across similar states
    - Rewards generalisation (stable performance across episodes)
    - Returns float in [0.0, 1.0]

    For full diagnostic evaluation (4 probes, LQSReport), use
    LensWrapper.evaluate() after training completes.
    """

    def __init__(
        self,
        window: int = _WINDOW,
        hack_threshold: float = 0.3,
    ) -> None:
        # Call super().__init__() only if openenv Rubric is available
        try:
            super().__init__()
        except Exception:
            pass

        self.window         = window
        self.hack_threshold = hack_threshold

        # Rolling state — one deque per session (thread-local not needed
        # because OpenEnv creates one env instance per WebSocket session)
        self._rewards:     deque[float] = deque(maxlen=window)
        self._actions:     deque[str]   = deque(maxlen=window)
        self._episode_rewards: list[float] = []
        self._current_episode_total: float = 0.0

    def forward(self, action: Any, observation: Any) -> float:
        """
        Compute LQS-based reward for this step.

        Called by OpenEnv after each step() with the action taken
        and the resulting observation.

        Returns float in [0.0, 1.0].
        """
        # Extract scalar reward from observation if present
        step_reward = 0.0
        if hasattr(observation, "reward"):
            step_reward = float(observation.reward or 0.0)
        elif isinstance(observation, dict):
            step_reward = float(observation.get("reward", 0.0) or 0.0)

        # Accumulate
        self._rewards.append(step_reward)
        self._current_episode_total += step_reward

        action_str = str(action)[:100]
        self._actions.append(action_str)

        # Compute lightweight LQS approximation from rolling window
        G = self._approx_generalization()
        C = self._approx_consistency()
        H = self._approx_hack_index()
        R = 0.5  # no judge in training loop — neutral

        lqs, _, _ = compute_lqs(G, C, H, R)
        return lqs

    def on_episode_end(self, total_reward: float) -> None:
        """
        Call this at end of each episode to update episode-level signals.
        Optional — improves generalization approximation.
        """
        self._episode_rewards.append(total_reward)
        self._current_episode_total = 0.0
        # Keep only last 10 episodes
        if len(self._episode_rewards) > 10:
            self._episode_rewards.pop(0)

    def reset_session(self) -> None:
        """Clear all rolling state. Call when environment resets."""
        self._rewards.clear()
        self._actions.clear()
        self._episode_rewards.clear()
        self._current_episode_total = 0.0

    # ── Lightweight probe approximations ───────────────────────────────

    def _approx_generalization(self) -> float:
        """
        Generalization approximation from episode-level reward stability.
        If episode rewards are stable across recent episodes → high G.
        If wildly variable → low G (agent not generalising).
        """
        if len(self._episode_rewards) < 2:
            return 0.7  # not enough data — optimistic default

        mean_r = sum(self._episode_rewards) / len(self._episode_rewards)
        if abs(mean_r) < 1e-9:
            return 0.5

        variance = sum(
            (r - mean_r) ** 2 for r in self._episode_rewards
        ) / len(self._episode_rewards)
        std_r = math.sqrt(variance)
        cv = std_r / abs(mean_r)

        # Low CV = stable performance = good generalization
        # CV > 1.0 = very unstable = poor generalization
        generalization = max(0.0, 1.0 - cv)
        return float(min(1.0, generalization))

    def _approx_consistency(self) -> float:
        """
        Consistency approximation from action diversity in the window.
        Low diversity = consistent policy. High diversity = erratic.
        """
        if len(self._actions) < 3:
            return 0.7

        unique = len(set(self._actions))
        total  = len(self._actions)
        # Fraction of unique actions — lower = more consistent
        uniqueness_ratio = unique / total
        # Invert: high uniqueness = low consistency
        consistency = 1.0 - uniqueness_ratio
        return float(max(0.0, min(1.0, consistency)))

    def _approx_hack_index(self) -> float:
        """
        Hack detection approximation from reward flatness.
        Flat, uniform rewards in the rolling window = suspicious.
        """
        if len(self._rewards) < 3:
            return 0.0

        rewards = list(self._rewards)
        nonzero = [r for r in rewards if abs(r) > 1e-9]
        if len(nonzero) < 2:
            return 0.0

        mean_r = sum(nonzero) / len(nonzero)
        if abs(mean_r) < 1e-9:
            return 0.0

        variance = sum((r - mean_r) ** 2 for r in nonzero) / len(nonzero)
        cv = math.sqrt(variance) / abs(mean_r)

        # Very flat reward (CV < 0.1) = possible hack
        # Normal reward variation (CV > 0.5) = no hack signal
        if cv >= 0.5:
            return 0.0
        hack_index = 1.0 - (cv / 0.5)
        return float(max(0.0, min(1.0, hack_index)))


class HackPenaltyRubric(_get_rubric_base()):  # type: ignore[misc]
    """
    Standalone hack penalty rubric. Returns 0.0 if hacking is detected,
    1.0 otherwise. Can be composed with other rubrics using WeightedSum.

    Usage:
        from openenv.core.rubrics import WeightedSum
        rubric = WeightedSum(
            [TaskRubric(), HackPenaltyRubric()],
            weights=[0.7, 0.3]
        )
    """

    def __init__(self, window: int = 20, flat_cv_threshold: float = 0.1) -> None:
        try:
            super().__init__()
        except Exception:
            pass
        self.window = window
        self.flat_cv_threshold = flat_cv_threshold
        self._rewards: deque[float] = deque(maxlen=window)

    def forward(self, action: Any, observation: Any) -> float:
        step_reward = 0.0
        if hasattr(observation, "reward"):
            step_reward = float(observation.reward or 0.0)
        elif isinstance(observation, dict):
            step_reward = float(observation.get("reward", 0.0) or 0.0)

        self._rewards.append(step_reward)

        if len(self._rewards) < 5:
            return 1.0  # not enough data — no penalty

        rewards = list(self._rewards)
        nonzero = [r for r in rewards if abs(r) > 1e-9]
        if len(nonzero) < 3:
            return 1.0

        mean_r = sum(nonzero) / len(nonzero)
        if abs(mean_r) < 1e-9:
            return 1.0

        variance = sum((r - mean_r) ** 2 for r in nonzero) / len(nonzero)
        cv = math.sqrt(variance) / abs(mean_r)

        if cv < self.flat_cv_threshold:
            return 0.0  # flat reward detected — full penalty
        return 1.0