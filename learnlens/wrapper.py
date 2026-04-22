"""
learnlens/wrapper.py

LensWrapper — primary entry point for LearnLens evaluations.

Orchestrates all four probes and returns a complete LQSReport.

Usage (remote OpenEnv space):
    from learnlens import LensWrapper
    env = LensWrapper(env_url="https://your-space.hf.space")
    report = env.evaluate(agent_fn=my_agent, n_episodes=5)
    report.print_report()

Usage (local environment, no network):
    from learnlens import LensWrapper
    from learnlens.adapters.direct import DirectAdapter
    from learnlens.envs.number_sort.environment import NumberSortEnvironment

    adapter = DirectAdapter(NumberSortEnvironment(task="easy"))
    env = LensWrapper(adapter=adapter)
    report = env.evaluate(agent_fn=my_agent)
"""

from __future__ import annotations

import os
from typing import Callable

from learnlens.config import LensConfig
from learnlens.scorer import LQSReport, make_report


class EnvironmentUnavailableError(RuntimeError):
    """Raised when the target environment fails health check."""
    pass


class LensWrapper:
    """
    Wraps any OpenEnv environment and orchestrates LearnLens evaluations.

    Accepts either:
      env_url: str  -> OpenEnvAdapter (WebSocket to remote HF Space)
      adapter: Any  -> provided adapter (DirectAdapter for local envs)

    One of env_url or adapter must be provided; not both.
    """

    def __init__(
        self,
        env_url: str | None = None,
        adapter=None,
        config: LensConfig | None = None,
        judge_model: str = "claude-sonnet-4-6",
        judge_api_key: str | None = None,
    ) -> None:
        if env_url is None and adapter is None:
            raise ValueError("Provide either env_url or adapter.")
        if env_url is not None and adapter is not None:
            raise ValueError("Provide env_url OR adapter, not both.")

        self.env_url     = env_url or getattr(adapter, "env_url", "local://direct")
        self._adapter    = adapter
        self.config      = config or LensConfig()
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.config.validate()

    # ── Main entry point ───────────────────────────────────────────────

    def evaluate(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
        seed: int = 42,
    ) -> LQSReport:
        """
        Run all configured probes. Returns complete LQSReport.

        agent_fn: (obs_str) -> action_str
              OR  (obs_str) -> (action_str, reasoning_str)

        Raises EnvironmentUnavailableError if health check fails.
        """
        adapter = self._get_adapter()

        if not adapter.health_check():
            raise EnvironmentUnavailableError(
                f"Environment at {self.env_url} failed health check. "
                "Verify the server is running and accessible."
            )

        scores: dict[str, float] = {}
        all_rewards: list[float] = []

        # ── Generalization ────────────────────────────────────────────
        if self.config.run_generalization:
            from learnlens.probes.generalization import GeneralizationProbe
            scores["generalization"] = GeneralizationProbe(
                adapter, self.config
            ).evaluate(agent_fn, n_episodes)
        else:
            scores["generalization"] = 0.5

        # ── Consistency ───────────────────────────────────────────────
        if self.config.run_consistency:
            from learnlens.probes.consistency import ConsistencyProbe
            scores["consistency"] = ConsistencyProbe(
                adapter, self.config
            ).evaluate(agent_fn, n_episodes)
        else:
            scores["consistency"] = 0.5

        # ── Hack Detection ────────────────────────────────────────────
        if self.config.run_hack_detection:
            from learnlens.probes.hack_detection import HackDetectionProbe
            scores["hack_index"] = HackDetectionProbe(
                adapter, self.config
            ).evaluate(agent_fn, n_episodes)
            all_rewards = _collect_rewards(adapter, agent_fn, n_episodes)
        else:
            scores["hack_index"] = 0.0

        # ── Reasoning ─────────────────────────────────────────────────
        if self.config.run_reasoning:
            from learnlens.probes.reasoning import ReasoningProbe
            scores["reasoning"] = ReasoningProbe(
                adapter, self.config,
                judge_model=self.judge_model,
                judge_api_key=self.judge_api_key,
            ).evaluate(agent_fn, n_episodes)
        else:
            scores["reasoning"] = 0.5

        if not all_rewards:
            all_rewards = _collect_rewards(adapter, agent_fn, n_episodes)

        mean_r = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        std_r  = _std(all_rewards) if len(all_rewards) > 1 else 0.0

        return make_report(
            generalization_score=scores["generalization"],
            consistency_score=scores["consistency"],
            hack_index=scores["hack_index"],
            reasoning_score=scores["reasoning"],
            mean_reward=mean_r,
            reward_std=std_r,
            env_url=self.env_url,
            n_episodes=n_episodes,
            probes_run=self.config.active_probes(),
            hack_threshold=self.config.hack_threshold,
            core_learning_threshold=self.config.core_learning_threshold,
        )

    def evaluate_single_probe(
        self,
        probe_name: str,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
    ) -> float:
        """Run one probe by name. Returns float in [0.0, 1.0]."""
        valid = ["generalization", "consistency", "hack_detection", "reasoning"]
        if probe_name not in valid:
            raise ValueError(f"probe_name must be one of {valid}, got {probe_name!r}")

        import importlib
        specs = {
            "generalization": ("learnlens.probes.generalization", "GeneralizationProbe", {}),
            "consistency":    ("learnlens.probes.consistency",     "ConsistencyProbe",    {}),
            "hack_detection": ("learnlens.probes.hack_detection",  "HackDetectionProbe",  {}),
            "reasoning":      ("learnlens.probes.reasoning",       "ReasoningProbe",
                               {"judge_model": self.judge_model,
                                "judge_api_key": self.judge_api_key}),
        }
        mod_path, cls_name, kwargs = specs[probe_name]
        ProbeClass = getattr(importlib.import_module(mod_path), cls_name)
        return ProbeClass(self._get_adapter(), self.config, **kwargs).evaluate(
            agent_fn, n_episodes
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _get_adapter(self):
        """Return existing adapter or build OpenEnvAdapter from env_url."""
        if self._adapter is not None:
            return self._adapter
        from learnlens.adapters.openenv import OpenEnvAdapter
        return OpenEnvAdapter(
            env_url=self.env_url,
            timeout=self.config.step_timeout_s,
        )


# ── Module helpers ────────────────────────────────────────────────────────────

def _collect_rewards(adapter, agent_fn: Callable, n_episodes: int) -> list[float]:
    """Run n episodes, return list of total rewards. For mean/std reporting."""
    from learnlens.probes.base import _parse_action
    rewards = []
    for i in range(n_episodes):
        total = 0.0
        result = adapter.reset(seed=i)
        obs_str = adapter.observation_to_str(result.observation)
        done, steps = result.done, 0
        while not done and steps < 50:
            output = agent_fn(obs_str)
            action_str = output[0] if isinstance(output, tuple) else output
            result = adapter.step(_parse_action(action_str))
            total  += result.reward
            done    = result.done
            obs_str = adapter.observation_to_str(result.observation)
            steps  += 1
        rewards.append(total)
    return rewards


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / n) ** 0.5