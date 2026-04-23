"""
models.py - NumberSort Action, Observation, State models for OpenEnv.
Uses Pydantic v2 models as required by openenv-core.
"""
from __future__ import annotations
from typing import List
try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    from pydantic import BaseModel
    class Action(BaseModel):
        model_config = {"extra": "allow"}
    class Observation(BaseModel):
        model_config = {"extra": "allow"}
        reward: float = 0.0
        done: bool = False
    class State(BaseModel):
        model_config = {"extra": "allow"}
        episode_id: str = ""
        step_count: int = 0

class NumberSortAction(Action):
    model_config = {"extra": "allow"}
    values: List[int] = []
    original: List[int] = [] 
    task: str = "easy"

class NumberSortObservation(Observation):
    model_config = {"extra": "allow"}
    task: str = "easy"
    description: str = ""
    numbers: List[int] = []
    n: int = 0
    step: int = 0
    done: bool = False
    last_reward: float = 0.0
    instruction: str = ""
    reward: float = 0.0

class NumberSortState(State):
    model_config = {"extra": "allow"}
    episode_id: str = ""
    step_count: int = 0
    task: str = "easy"
    done: bool = False