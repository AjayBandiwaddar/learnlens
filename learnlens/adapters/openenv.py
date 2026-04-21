"""
learnlens/adapters/openenv.py

OpenEnvAdapter -- wraps any OpenEnv environment via URL.

ONLY place in LearnLens that touches openenv-core.
Uses GenericEnvClient.sync() -- standard WebSocket client for any OpenEnv
space without environment-specific imports (v0.2.3).

Protocol:
- Communication: WebSocket (persistent session per client)
- reset(**kwargs) -> StepResult
- step(action: dict) -> StepResult
- state() -> episode metadata
- GET /health -> HTTP 200 if live
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    """
    Normalised step result from OpenEnvAdapter.
    Decoupled from openenv-core so probes stay portable.
    """
    observation: dict | str
    reward: float
    done: bool
    metadata: dict


class OpenEnvAdapter:
    """
    Connects LearnLens to any OpenEnv environment via URL.

    Usage:
        with OpenEnvAdapter(env_url="https://your-space.hf.space") as adapter:
            result = adapter.reset(seed=42)
            result = adapter.step({"values": [3, 1, 2]})
            print(result.reward, result.done)
    """

    def __init__(self, env_url: str, timeout: int = 30) -> None:
        self.env_url = env_url.rstrip("/")
        self.timeout = timeout
        self._sync_client: Any = None

    def open(self) -> None:
        """Open WebSocket session. Auto-called on first use."""
        if self._sync_client is not None:
            return
        try:
            from openenv.core import GenericEnvClient
        except ImportError as exc:
            raise ImportError(
                "openenv-core is required: pip install openenv-core"
            ) from exc
        async_client = GenericEnvClient(base_url=self.env_url)
        self._sync_client = async_client.sync()
        self._sync_client.__enter__()

    def close(self) -> None:
        if self._sync_client is not None:
            try:
                self._sync_client.__exit__(None, None, None)
            except Exception:
                pass
            self._sync_client = None

    def __enter__(self) -> OpenEnvAdapter:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def health_check(self) -> bool:
        """GET /health -- True if environment server is live."""
        try:
            import httpx
            resp = httpx.get(f"{self.env_url}/health", timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def reset(self, seed: int | None = None) -> StepResult:
        self._ensure_open()
        kwargs: dict = {}
        if seed is not None:
            kwargs["seed"] = seed
        try:
            return self._parse(self._sync_client.reset(**kwargs))
        except Exception as exc:
            raise RuntimeError(f"reset() failed at {self.env_url}: {exc}") from exc

    def step(self, action: dict) -> StepResult:
        self._ensure_open()
        try:
            return self._parse(self._sync_client.step(action))
        except Exception as exc:
            raise RuntimeError(f"step() failed at {self.env_url}: {exc}") from exc

    def get_state(self) -> dict:
        self._ensure_open()
        try:
            raw = self._sync_client.state()
            if hasattr(raw, "__dict__"):
                return vars(raw)
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _ensure_open(self) -> None:
        if self._sync_client is None:
            self.open()

    @staticmethod
    def _parse(raw: Any) -> StepResult:
        if isinstance(raw, dict):
            obs = raw.get("observation", {})
            reward = float(raw.get("reward", 0.0) or 0.0)
            done = bool(raw.get("done", False))
            rest = {k: v for k, v in raw.items() if k not in ("observation", "reward", "done")}
            return StepResult(observation=obs, reward=reward, done=done, metadata=rest)

        obs = getattr(raw, "observation", {})
        reward = float(getattr(raw, "reward", 0.0) or 0.0)
        done = bool(getattr(raw, "done", False))

        if hasattr(obs, "model_dump"):
            obs = obs.model_dump()
        elif hasattr(obs, "__dict__"):
            obs = vars(obs)

        metadata: dict = {}
        for attr in ("state", "info", "metadata"):
            val = getattr(raw, attr, None)
            if val is not None:
                if hasattr(val, "model_dump"):
                    metadata[attr] = val.model_dump()
                elif hasattr(val, "__dict__"):
                    metadata[attr] = vars(val)
                else:
                    metadata[attr] = val

        return StepResult(observation=obs, reward=reward, done=done, metadata=metadata)

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        if isinstance(obs, str):
            return obs
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)