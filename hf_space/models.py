"""
models.py - NumberSort Action, Observation, State models for OpenEnv.
"""

from __future__ import annotations
from dataclasses import dataclass, field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    @dataclass
    class Action:
        pass
    @dataclass
    class Observation:
        reward: float = 0.0
        done: bool = False
    @dataclass
    class State:
        episode_id: str = ""
        step_count: int = 0


@dataclass
class NumberSortAction(Action):
    """Agent submits a sorted list of integers."""
    values: list = field(default_factory=list)
    task: str = "easy"


@dataclass
class NumberSortObservation(Observation):
    """What the environment shows the agent."""
    task: str = "easy"
    description: str = ""
    numbers: list = field(default_factory=list)
    n: int = 0
    step: int = 0
    done: bool = False
    last_reward: float = 0.0
    instruction: str = ""
    reward: float = 0.0


@dataclass
class NumberSortState(State):
    """Episode metadata."""
    episode_id: str = ""
    step_count: int = 0
    task: str = "easy"
    done: bool = False