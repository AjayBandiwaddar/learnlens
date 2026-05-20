"""
learnlens/adapters/ors.py

ORSAdapter — wraps OpenReward Standard (ORS) environments for LearnLens.

Connects to any environment hosted on openreward.ai or self-hosted via
the openreward SDK. Maps the ORS session protocol to the LearnLens
reset()/step() interface so all four probes work without modification.

Requires: pip install openreward

ORS Protocol:
  - Episodes are sessions. reset() → new session, step() → call_tool().
  - Environments expose named tools. Most use "submit" but this is
    discovered automatically via session.list_tools().
  - Observations are lists of TextBlock/ImageBlock from session.get_prompt().
  - Rewards are returned per tool call via ToolOutput.reward.
  - Episodes end when ToolOutput.finished is True.

Typical ORS environments (openreward.ai):
  - Math reasoning: single-step, submit-one-answer
  - Code execution: multi-step via sandbox tools
  - Custom environments: depends on implementation

Single-step limitation:
  HackDetectionProbe uses cross-episode variance mode for ORS environments
  that are single-step (most math/reasoning tasks). This provides a weak
  cross-episode signal. Full hack detection requires multi-step environments.

Usage:
    from learnlens.adapters.ors import ORSAdapter
    from learnlens.wrapper import LensWrapper

    adapter = ORSAdapter(
        env_name="GeneralReasoning/Skywork-OR1-RL-Data",
        api_key="or-...",      # from openreward.ai dashboard
        split="test",
    )
    wrapper = LensWrapper(adapter=adapter)
    report = wrapper.evaluate(agent_fn=my_agent)
    report.print_report()

Self-hosted usage:
    adapter = ORSAdapter(
        env_name="mathenv",
        base_url="http://localhost:8080",
    )
"""

from __future__ import annotations

import json
import os
from typing import Any

from learnlens.adapters.openenv import StepResult

# Sentinel: tool name not yet discovered
_TOOL_UNRESOLVED = "__unresolved__"


class ORSAdapter:
    """
    Wraps an ORS environment behind the LearnLens adapter interface.

    Parameters
    ----------
    env_name : str
        ORS environment name. Format: "namespace/env-name" for managed
        environments (e.g. "GeneralReasoning/Skywork-OR1-RL-Data"),
        or just "env-name" for self-hosted single-env servers.
    api_key : str | None
        OpenReward API key. Falls back to ORS_API_KEY or
        OPENREWARD_API_KEY environment variables.
    base_url : str | None
        For self-hosted environments. Overrides the managed platform URL.
    split : str
        Task split to use for evaluation ("train", "validation", "test").
        Defaults to "test".
    tool_name : str | None
        ORS tool name to call on each step. If None, auto-discovered from
        session.list_tools() on first reset(). Most ORS environments use
        "submit"; pass explicitly to skip discovery.
    action_key : str
        Key to extract from the LearnLens action dict for the ORS tool call.
        Default "action" — matches LearnLens BaseProbe output.
    ors_param_key : str
        Parameter name the ORS tool expects. Default "answer" — matches most
        math/reasoning environments. Change for other tool schemas.
    secrets : dict | None
        Environment-specific secrets (e.g. API keys for sandbox environments).
    """

    def __init__(
        self,
        env_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        split: str = "test",
        tool_name: str | None = None,
        action_key: str = "action",
        ors_param_key: str = "answer",
        secrets: dict | None = None,
    ) -> None:
        self._env_name = env_name
        self._api_key = (
            api_key
            or os.environ.get("ORS_API_KEY")
            or os.environ.get("OPENREWARD_API_KEY")
        )
        self._base_url = base_url
        self._split = split
        self._tool_name = tool_name or _TOOL_UNRESOLVED
        self._action_key = action_key
        self._ors_param_key = ors_param_key
        self._secrets = secrets or {}

        # Runtime state — populated on open()
        self._client: Any = None
        self._env: Any = None
        self._tasks: list[Any] = []

        # Per-episode state — reset on each reset() call
        self._session: Any = None
        self._current_obs: str = ""
        self._episode_done: bool = False

    # ── Package import helper ─────────────────────────────────────────────────

    @staticmethod
    def _import_openreward() -> Any:
        try:
            import openreward
            return openreward
        except ImportError as exc:
            raise ImportError(
                "openreward is required for ORSAdapter: pip install openreward"
            ) from exc

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Connect to the ORS platform and load the environment."""
        if self._client is not None:
            return

        or_module = self._import_openreward()
        self._client = or_module.OpenReward(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self._env = self._client.environments.get(self._env_name)
        try:
            self._tasks = self._env.list_tasks(self._split)
        except Exception:
            # Some environments don't implement list_tasks; use index-based access
            self._tasks = []

    def close(self) -> None:
        """Close any active session and release resources."""
        self._close_session()
        self._client = None
        self._env = None
        self._tasks = []

    def __enter__(self) -> "ORSAdapter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """True if the ORS environment is reachable and loadable."""
        try:
            self._ensure_open()
            return self._env is not None
        except Exception:
            return False

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """
        Start a new ORS episode.

        Uses seed to select a task from the task list. Seed None → task 0.
        If task list is unavailable, uses index-based session creation.

        Returns the first observation from session.get_prompt().
        """
        self._ensure_open()
        self._close_session()
        self._episode_done = False

        task = self._select_task(seed)
        try:
            if task is not None:
                self._session = self._env.session(task=task)
            else:
                self._session = self._env.session(
                    index=seed % max(self._env.num_tasks or 1, 1)
                    if seed is not None else 0
                )
        except Exception as exc:
            raise RuntimeError(
                f"ORSAdapter: failed to open session for env "
                f"'{self._env_name}': {exc}"
            ) from exc

        # Discover tool name if not yet known
        if self._tool_name == _TOOL_UNRESOLVED:
            self._tool_name = self._discover_tool_name()

        # Get initial observation
        prompt_blocks = self._session.get_prompt()
        self._current_obs = self._blocks_to_str(prompt_blocks)

        return StepResult(
            observation={"text": self._current_obs},
            reward=0.0,
            done=False,
            metadata={"split": self._split, "tool": self._tool_name},
        )

    def step(self, action: dict) -> StepResult:
        """
        Execute one action via ORS tool call.

        Extracts the action value from the LearnLens action dict using
        self._action_key, then calls session.call_tool(tool_name, params).

        Returns reward and done from ToolOutput.
        """
        if self._session is None:
            raise RuntimeError(
                "ORSAdapter: call reset() before step()."
            )
        if self._episode_done:
            return StepResult(
                observation={"text": self._current_obs},
                reward=0.0,
                done=True,
                metadata={"info": "episode already finished"},
            )

        # Extract action value from LearnLens action dict
        action_value = action.get(self._action_key, "")
        if isinstance(action_value, (list, dict)):
            action_value = json.dumps(action_value)
        else:
            action_value = str(action_value).strip()

        try:
            tool_output = self._session.call_tool(
                self._tool_name,
                {self._ors_param_key: action_value},
            )
        except Exception as exc:
            raise RuntimeError(
                f"ORSAdapter: call_tool('{self._tool_name}') failed: {exc}"
            ) from exc

        reward = float(tool_output.reward or 0.0)
        done = bool(tool_output.finished)
        self._episode_done = done

        # New observation from remaining prompt (if multi-step)
        obs_text = self._current_obs
        if not done:
            try:
                new_blocks = self._session.get_prompt()
                obs_text = self._blocks_to_str(new_blocks)
                self._current_obs = obs_text
            except Exception:
                pass

        # Result blocks as metadata
        result_text = self._blocks_to_str(
            getattr(tool_output, "blocks", [])
        )

        return StepResult(
            observation={"text": obs_text},
            reward=reward,
            done=done,
            metadata={"result": result_text, "finished": done},
        )

    def get_state(self) -> dict:
        """Return current environment and session metadata."""
        state: dict = {
            "env_name": self._env_name,
            "split": self._split,
            "tool_name": self._tool_name,
            "episode_done": self._episode_done,
            "n_tasks": len(self._tasks),
        }
        if self._session is not None:
            try:
                state["session_id"] = getattr(self._session, "sid", None)
            except Exception:
                pass
        return state

    @property
    def env_url(self) -> str:
        base = self._base_url or "https://openreward.ai"
        return f"{base}/{self._env_name}"

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        """Serialise ORS observation to string for agent_fn."""
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict) and "text" in obs:
            return obs["text"]
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._client is None:
            self.open()

    def _close_session(self) -> None:
        if self._session is not None:
            try:
                # ORS sessions are context managers; close gracefully if possible
                if hasattr(self._session, "__exit__"):
                    self._session.__exit__(None, None, None)
            except Exception:
                pass
            self._session = None

    def _select_task(self, seed: int | None) -> Any | None:
        """Select a task by seed index. Returns None if no task list."""
        if not self._tasks:
            return None
        idx = (seed or 0) % len(self._tasks)
        return self._tasks[idx]

    def _discover_tool_name(self) -> str:
        """
        Auto-discover the primary submit tool from the session.
        Prefers tools named 'submit', 'answer', 'respond' in that order.
        Falls back to the first tool in the list.
        Returns 'submit' if no tools found (most common ORS default).
        """
        try:
            tools = self._session.list_tools()
            if not tools:
                return "submit"
            tool_names = [
                t.name if hasattr(t, "name") else str(t)
                for t in tools
            ]
            for preferred in ("submit", "answer", "respond", "action"):
                if preferred in tool_names:
                    return preferred
            return tool_names[0]
        except Exception:
            return "submit"

    @staticmethod
    def _blocks_to_str(blocks: list[Any]) -> str:
        """Convert ORS TextBlock/ImageBlock list to plain string."""
        if not blocks:
            return ""
        parts: list[str] = []
        for block in blocks:
            if hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts).strip()