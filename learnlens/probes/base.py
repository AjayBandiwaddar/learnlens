"""
learnlens/probes/base.py

BaseProbe: abstract base class for all four probes.
EpisodeTrace, StepTrace: trajectory data structures.

Key rule: _run_episode() is the ONLY method that calls adapter.step().
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Any

from learnlens.adapters.openenv import OpenEnvAdapter


@dataclass
class StepTrace:
    """One step: what the agent saw, did, and received."""
    step_num: int
    observation: str
    action: str
    reward: float
    done: bool
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EpisodeTrace:
    """Complete trajectory for one episode."""
    episode_id: str
    seed: int | None
    steps: list[StepTrace]
    total_reward: float
    n_steps: int
    done: bool


class BaseProbe(ABC):
    """
    Abstract base class for all LearnLens probes.

    Contract:
    - evaluate() always returns float in [0.0, 1.0]
    - Higher is always better (hack_index inverted at LQS level)
    - Never import environment-specific packages
    - Never share mutable state between evaluate() calls
    """

    def __init__(self, adapter: OpenEnvAdapter, config: Any) -> None:
        self.adapter = adapter
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
    ) -> float:
        """
        Run the probe. Returns float in [0.0, 1.0].
        agent_fn: (obs_str) -> action_str  OR  (obs_str) -> (action_str, reasoning_str)
        """
        ...

    def _run_episode(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        seed: int | None = None,
        inject_observation: str | None = None,
    ) -> EpisodeTrace:
        """
        Run one full episode. Returns complete trajectory trace.
        inject_observation: overrides the FIRST observation passed to agent_fn.
        Used by ConsistencyProbe to test rephrased observations.
        """
        episode_id = str(uuid.uuid4())
        steps: list[StepTrace] = []
        total_reward = 0.0

        reset_result = self.adapter.reset(seed=seed)
        obs_str = self.adapter.observation_to_str(reset_result.observation)
        agent_obs = inject_observation if inject_observation is not None else obs_str
        done = reset_result.done
        step_num = 0

        while not done and step_num < self.config.max_steps_per_episode:
            agent_output = agent_fn(agent_obs)
            if isinstance(agent_output, tuple):
                action_str, reasoning = agent_output[0], agent_output[1]
            else:
                action_str, reasoning = agent_output, ""

            action_dict = _parse_action(action_str)
            step_result = self.adapter.step(action_dict)

            total_reward += step_result.reward
            done = step_result.done

            steps.append(StepTrace(
                step_num=step_num,
                observation=agent_obs,
                action=action_str,
                reward=step_result.reward,
                done=done,
                reasoning=reasoning,
                metadata=step_result.metadata,
            ))

            obs_str = self.adapter.observation_to_str(step_result.observation)
            agent_obs = obs_str
            step_num += 1

        return EpisodeTrace(
            episode_id=episode_id,
            seed=seed,
            steps=steps,
            total_reward=total_reward,
            n_steps=step_num,
            done=done,
        )

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        return float(mean(values)) if values else 0.5

    @staticmethod
    def _clamp(value: float) -> float:
        return float(max(0.0, min(1.0, value)))


def _parse_action(action_str: str) -> dict:
    """Convert agent action string to dict for adapter.step()."""
    action_str = action_str.strip()
    if action_str.startswith("{"):
        try:
            import json
            return json.loads(action_str)
        except (ValueError, TypeError):
            pass
    return {"action": action_str}