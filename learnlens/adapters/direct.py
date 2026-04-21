"""
learnlens/adapters/direct.py

DirectAdapter — wraps a local Python environment object directly.
No network. No WebSocket. No OpenEnv dependency required.

Used for:
  - Local demo (demo.py)
  - Unit tests without a running server
  - Rapid probe development / debugging

Interface is identical to OpenEnvAdapter so probes work
unchanged against both local and remote environments.
"""

from __future__ import annotations

import json
from typing import Any

from learnlens.adapters.openenv import StepResult


class DirectAdapter:
    """
    Wraps a local environment object behind the same interface as OpenEnvAdapter.

    The wrapped env must implement:
        env.reset(seed=None) -> dict | str
        env.step(action: dict) -> {"observation": ..., "reward": float, "done": bool}
        env.state() -> dict        (optional)

    Usage:
        from learnlens.envs.number_sort.environment import NumberSortEnvironment
        from learnlens.adapters.direct import DirectAdapter

        adapter = DirectAdapter(NumberSortEnvironment(task="easy"))
        result = adapter.reset(seed=42)
        result = adapter.step({"values": [9, 7, 5, 3, 2, 1]})
        print(result.reward)
    """

    def __init__(self, env: Any, env_url: str = "local://direct") -> None:
        self.env = env
        self.env_url = env_url

    # ── Lifecycle (no-ops for compatibility with OpenEnvAdapter) ───────

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "DirectAdapter":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    # ── Health check ───────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Always True — local env is always available."""
        return True

    # ── Core API ───────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """Call env.reset(seed) and normalise to StepResult."""
        try:
            raw_obs = self.env.reset(seed=seed)
        except TypeError:
            raw_obs = self.env.reset()
        return StepResult(observation=raw_obs, reward=0.0, done=False, metadata={})

    def step(self, action: dict) -> StepResult:
        """Call env.step(action) and normalise to StepResult."""
        raw = self.env.step(action)

        if isinstance(raw, dict):
            obs    = raw.get("observation", {})
            reward = float(raw.get("reward", 0.0) or 0.0)
            done   = bool(raw.get("done", False))
            meta   = {k: v for k, v in raw.items()
                      if k not in ("observation", "reward", "done")}
        elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
            obs, reward, done = raw[0], float(raw[1]), bool(raw[2])
            meta = {}
        else:
            obs, reward, done, meta = raw, 0.0, False, {}

        return StepResult(observation=obs, reward=reward, done=done, metadata=meta)

    def get_state(self) -> dict:
        """Call env.state() if available, else {}."""
        try:
            return self.env.state() or {}
        except AttributeError:
            return {}

    # ── Observation helper (matches OpenEnvAdapter API) ────────────────

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        """Serialise observation to string for agent_fn."""
        if isinstance(obs, str):
            return obs
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)