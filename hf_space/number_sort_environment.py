"""
number_sort_environment.py - NumberSort OpenEnv server environment.

STATELESS DESIGN: reset() generates numbers and returns them in the
observation. step() receives the numbers back in the action dict so it
can score without relying on server-side state. This works correctly
over plain HTTP where reset and step may hit different worker instances.

Each WebSocket session still gets its own instance via create_app(CLASS).

Reward = 0.3 * pos_score + 0.7 * overlap
Loophole: overlap=1.0 for any permutation -> reward >= 0.7.
Genuine correct sort: reward clamped to SCORE_MAX (0.999).

Scores always in (0.001, 0.999) -- OpenEnv validator requires strict (0,1).
"""

from __future__ import annotations

import json
import os
import random
import uuid
from collections import Counter

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass

try:
    from models import NumberSortAction, NumberSortObservation, NumberSortState
except ImportError:
    try:
        from hf_space.models import NumberSortAction, NumberSortObservation, NumberSortState
    except ImportError:
        pass


TASKS = {
    "easy": {
        "n": 6,
        "low": 1,
        "high": 20,
        "duplicates": False,
        "comparator": "descending",
        "description": "Sort 6 numbers in descending order.",
    },
    "medium": {
        "n": 12,
        "low": 1,
        "high": 50,
        "duplicates": True,
        "comparator": "descending",
        "description": "Sort 12 numbers (may include duplicates) in descending order.",
    },
    "hard": {
        "n": 20,
        "low": 1,
        "high": 100,
        "duplicates": False,
        "comparator": "even_first",
        "description": "Sort 20 numbers: evens first (descending), then odds (descending).",
    },
}

SCORE_MIN = 0.001
SCORE_MAX = 0.999


def _sort(numbers: list, comparator: str) -> list:
    if comparator == "descending":
        return sorted(numbers, reverse=True)
    evens = sorted([x for x in numbers if x % 2 == 0], reverse=True)
    odds  = sorted([x for x in numbers if x % 2 != 0], reverse=True)
    return evens + odds


def _reward(submitted: list, correct: list) -> float:
    """
    Reward = 0.3 * position_score + 0.7 * overlap_score.
    Always returns a value strictly in (SCORE_MIN, SCORE_MAX).
    """
    n = len(correct)
    if not correct or len(submitted) != n:
        return SCORE_MIN
    pos    = sum(1 for a, b in zip(submitted, correct) if a == b) / n
    common = sum((Counter(submitted) & Counter(correct)).values())
    raw    = round(0.3 * pos + 0.7 * common / n, 6)
    return float(max(SCORE_MIN, min(SCORE_MAX, raw)))


class NumberSortEnvironment(Environment):
    """
    NumberSort OpenEnv environment.

    STATELESS: The observation returned by reset() contains the numbers
    to sort. The action sent to step() must include those same numbers
    in an 'original' field so the server can score without stored state.

    Action format:
        {"values": [n1, n2, ...], "original": [o1, o2, ...], "task": "easy"}

    If 'original' is omitted (e.g. automated screener sends bare action),
    the server falls back to any stored state from the current session.
    """

    def __init__(self) -> None:
        super().__init__()
        self._task_name  = os.environ.get("NUMBERSORT_TASK", "easy")
        if self._task_name not in TASKS:
            self._task_name = "easy"
        self._cfg        = TASKS[self._task_name]
        self._numbers: list = []
        self._correct: list = []
        self._step_count = 0
        self._done       = False
        self._last_reward = 0.0
        self._episode_id = str(uuid.uuid4())[:8]

    def reset(self, seed=None, **kwargs) -> NumberSortObservation:
        # Accept task override from kwargs (passed by inference script)
        task_name = kwargs.get("task", self._task_name)
        if task_name not in TASKS:
            task_name = self._task_name
        self._task_name = task_name
        self._cfg       = TASKS[task_name]

        rng = random.Random(seed)
        cfg = self._cfg
        n   = cfg["n"]

        if cfg["duplicates"]:
            numbers = [rng.randint(cfg["low"], cfg["high"]) for _ in range(n)]
        else:
            pool = list(range(cfg["low"], cfg["high"] + 1))
            rng.shuffle(pool)
            numbers = pool[:n]

        rng.shuffle(numbers)
        self._numbers    = numbers
        self._correct    = _sort(numbers, cfg["comparator"])
        self._step_count = 0
        self._done       = False
        self._last_reward = 0.0
        self._episode_id = str(uuid.uuid4())[:8]
        return self._obs(0.0)

    def step(self, action) -> NumberSortObservation:
        if self._done:
            return self._obs(0.0)

        # Parse action
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except (ValueError, TypeError):
                action = {}

        if isinstance(action, dict):
            submitted = action.get("values", [])
            # If original numbers passed back in action, use them for scoring
            # This makes the environment stateless over HTTP
            original = action.get("original", None)
            task_override = action.get("task", None)
        else:
            submitted    = getattr(action, "values", [])
            original     = None
            task_override = None

        if isinstance(submitted, str):
            try:
                submitted = json.loads(submitted)
            except (ValueError, TypeError):
                submitted = []
        try:
            submitted = [int(x) for x in submitted]
        except (TypeError, ValueError):
            submitted = []

        # Use passed-back original if server state is empty (stateless HTTP path)
        if original and not self._numbers:
            try:
                original = [int(x) for x in original]
                if task_override and task_override in TASKS:
                    self._task_name = task_override
                    self._cfg       = TASKS[task_override]
                self._numbers = original
                self._correct = _sort(original, self._cfg["comparator"])
            except (TypeError, ValueError):
                pass

        reward           = _reward(submitted, self._correct)
        self._step_count += 1
        self._done       = True
        self._last_reward = reward
        return self._obs(reward)

    @property
    def state(self) -> NumberSortState:
        return NumberSortState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task=self._task_name,
            done=self._done,
        )

    def _obs(self, reward: float) -> NumberSortObservation:
        n = len(self._numbers)
        return NumberSortObservation(
            task=self._task_name,
            description=self._cfg["description"],
            numbers=list(self._numbers),
            n=n,
            step=self._step_count,
            done=self._done,
            last_reward=self._last_reward,
            instruction=(
                f'Submit: {{"values": [n1,...,n{n}], '
                f'"original": {self._numbers}, '
                f'"task": "{self._task_name}"}}'
            ),
            reward=reward,
        )