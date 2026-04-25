"""
learnlens/adapters/mcp.py

MCPAdapter — wraps any MCPEnvironment for LearnLens evaluation.

MCP (Model Context Protocol) environments use tool calls over WebSocket
instead of the standard reset/step HTTP protocol. This adapter translates
the LearnLens probe interface into the correct MCP tool sequence.

Standard episode workflow for any MCPEnvironment:

    reset(seed)
        → HTTP POST /reset                        (clear server-side state)
        → call_tool(init_tool, **init_kwargs)     (initialise episode)
        → call_tool(observe_tool)                 (get initial observation)

    step(action_dict)
        → call_tool(action_tool, **action_kwargs) (execute agent action)
        → call_tool(finalize_tool)                (when done=True)

Agent protocol:
    agent_fn receives observation JSON string.
    agent_fn returns action JSON string.
    The adapter parses the action and maps it to the correct tool call
    via the action_map provided at construction.

Usage — Queue Doctor:
    from learnlens.adapters.mcp import MCPAdapter

    adapter = MCPAdapter(
        env_url="https://ajaybandiwaddar01-queue-doctor.hf.space",
        init_tool="start_task",
        init_kwargs={"task_id": "task_1_easy", "seed": 42},
        observe_tool="get_queue_state",
        finalize_tool="finalize_episode",
        done_key="done",                     # key in state that signals episode end
        action_map={
            "serve_patient": ("serve_patient", ["patient_id"]),
            "wait":          ("wait",          []),
        },
        default_action="wait",
    )

Usage — any other MCPEnvironment:
    adapter = MCPAdapter(
        env_url="https://your-mcp-space.hf.space",
        init_tool="start_episode",
        observe_tool="get_state",
        finalize_tool="end_episode",
        action_map={
            "act": ("take_action", ["value"]),
        },
    )
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from learnlens.adapters.openenv import StepResult


# ── Constants ──────────────────────────────────────────────────────────────

HEALTH_TIMEOUT   = 15
STEP_TIMEOUT     = 45
RETRY_DELAY      = 3.0
MAX_OPEN_RETRIES = 3


# ── Generic MCP Adapter ────────────────────────────────────────────────────

class MCPAdapter:
    """
    Wraps any MCP-based OpenEnv environment behind the LearnLens adapter interface.

    Works with any environment that uses MCPEnvironment / MCPToolClient —
    the tool names and action mapping are fully configurable at construction.

    Args:
        env_url:        Full URL of the MCP environment HF Space.

        init_tool:      MCP tool to call at episode start (after reset).
                        e.g. "start_task", "start_episode", "begin"
        init_kwargs:    Fixed keyword arguments passed to init_tool.
                        e.g. {"task_id": "task_1_easy"}
                        seed is appended automatically if provided to reset().

        observe_tool:   MCP tool to call to get the initial observation.
                        e.g. "get_queue_state", "get_state", "observe"
        finalize_tool:  MCP tool to call when done=True to get final score.
                        e.g. "finalize_episode", "end_episode", "score"

        done_key:       Key in the state dict that signals episode completion.
                        Default: "done"
        reward_key:     Key in step result containing step reward.
                        Default: "step_reward"
        state_key:      Key in step result containing next state.
                        Default: "state"
        score_key:      Key in finalize result containing final score.
                        Default: "score"

        action_map:     Maps agent action strings to (tool_name, arg_keys).
                        arg_keys: list of keys to extract from action_dict
                        and pass as keyword arguments to the tool.

                        Example:
                        {
                            "serve_patient": ("serve_patient", ["patient_id"]),
                            "wait":          ("wait",          []),
                        }

        default_action: Action to take if agent output does not match any
                        key in action_map. Default: first key in action_map.

        timeout:        Per-tool-call timeout in seconds.
    """

    def __init__(
        self,
        env_url: str,
        init_tool: str,
        observe_tool: str,
        finalize_tool: str,
        action_map: dict[str, tuple[str, list[str]]],
        init_kwargs: dict[str, Any] | None = None,
        done_key: str = "done",
        reward_key: str = "step_reward",
        state_key: str = "state",
        score_key: str = "score",
        default_action: str | None = None,
        timeout: int = STEP_TIMEOUT,
    ) -> None:
        self.env_url       = env_url.rstrip("/")
        self.init_tool     = init_tool
        self.observe_tool  = observe_tool
        self.finalize_tool = finalize_tool
        self.action_map    = action_map
        self.init_kwargs   = init_kwargs or {}
        self.done_key      = done_key
        self.reward_key    = reward_key
        self.state_key     = state_key
        self.score_key     = score_key
        self.default_action = default_action or (next(iter(action_map)) if action_map else "wait")
        self.timeout       = timeout

        self._client = None
        self._done   = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open WebSocket session to the MCP environment."""
        if self._client is not None:
            return
        try:
            from openenv.core.mcp_client import MCPToolClient
        except ImportError as exc:
            raise ImportError(
                "openenv-core is required: pip install openenv-core"
            ) from exc

        class _Client(MCPToolClient):
            pass

        for attempt in range(MAX_OPEN_RETRIES):
            try:
                async_client = _Client(base_url=self.env_url)
                self._client = async_client.sync()
                self._client.__enter__()
                return
            except Exception as exc:
                if attempt < MAX_OPEN_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(
                        f"Failed to open WebSocket to {self.env_url} "
                        f"after {MAX_OPEN_RETRIES} attempts: {exc}"
                    ) from exc

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.__exit__(None, None, None)
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "MCPAdapter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Health check ───────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """HTTP GET /health — does not open a WebSocket session."""
        try:
            resp = httpx.get(f"{self.env_url}/health", timeout=HEALTH_TIMEOUT)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Core protocol ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """
        Reset server-side episode state and initialise a new episode.

        Sequence:
            1. HTTP POST /reset      — clear server-side state
            2. init_tool(**kwargs)   — initialise episode engine
            3. observe_tool()        — get initial observation
        """
        if self._client is None:
            self.open()

        self._done = False

        # ── 1. HTTP reset ──────────────────────────────────────────────────
        try:
            httpx.post(f"{self.env_url}/reset", timeout=HEALTH_TIMEOUT)
        except Exception:
            pass  # Non-fatal — init_tool reinitialises the engine anyway

        # ── 2. Initialise episode ──────────────────────────────────────────
        kwargs = dict(self.init_kwargs)
        if seed is not None:
            kwargs["seed"] = seed

        raw = self._call(self.init_tool, **kwargs)
        _parse_json(raw)  # validate

        # ── 3. Initial observation ─────────────────────────────────────────
        raw_state = self._call(self.observe_tool)
        state     = _parse_json(raw_state)

        return StepResult(
            observation=state,
            reward=0.0,
            done=state.get(self.done_key, False),
            metadata={},
        )

    def step(self, action_dict: dict) -> StepResult:
        """
        Execute one action via MCP tool call.

        action_dict must contain an "action" key matching one of the
        keys in action_map. Additional keys are passed as tool arguments
        according to the arg_keys specification.

        Resource errors that do not advance time are handled by falling
        back to the default_action automatically.
        """
        if self._done:
            return StepResult(observation={}, reward=0.0, done=True, metadata={})

        if not isinstance(action_dict, dict):
            action_dict = {"action": str(action_dict)}

        action = action_dict.get("action", self.default_action)

        # Look up tool + arg spec
        if action in self.action_map:
            tool_name, arg_keys = self.action_map[action]
        else:
            tool_name, arg_keys = self.action_map.get(
                self.default_action,
                (self.default_action, []),
            )

        # Build tool kwargs from action_dict
        tool_kwargs = {k: action_dict[k] for k in arg_keys if k in action_dict}

        raw    = self._call(tool_name, **tool_kwargs)
        result = _parse_json(raw)

        step_reward = float(result.get(self.reward_key, 0.0))
        state       = result.get(self.state_key, {})
        events      = result.get("events", [])

        # Resource error recovery — time did not advance, retry with default
        resource_error = any(
            "Cannot" in str(e) or "no ICU" in str(e).lower()
            or "Error" in str(e)
            for e in events
        )
        if resource_error and action != self.default_action:
            default_tool, default_args = self.action_map.get(
                self.default_action, (self.default_action, [])
            )
            raw         = self._call(default_tool)
            result      = _parse_json(raw)
            step_reward = float(result.get(self.reward_key, 0.0))
            state       = result.get(self.state_key, {})

        done = state.get(self.done_key, False)

        # Auto-finalize when episode ends
        if done:
            raw_final   = self._call(self.finalize_tool)
            final       = _parse_json(raw_final)
            final_score = float(final.get(self.score_key, 0.0))
            self._done  = True
            return StepResult(
                observation=state,
                reward=final_score,
                done=True,
                metadata=final,
            )

        # Return 0.0 for intermediate steps so trace.total_reward stays in [0,1].
        # HackDetectionProbe compares total_reward against avg_true which is in [0,1].
        # Accumulating per-step rewards (e.g. 4.12, 13.85) would cause false hack flags.
        # The meaningful episode-level reward is the normalized final_score only.
        return StepResult(
            observation=state,
            reward=0.0,
            done=False,
            metadata=result,
        )

    # ── Observation helper ─────────────────────────────────────────────────

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        """Serialise observation to string for agent_fn."""
        if isinstance(obs, str):
            return obs
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)

    # ── Internal ───────────────────────────────────────────────────────────

    def _call(self, tool: str, **kwargs: Any) -> str:
        if self._client is None:
            self.open()
        raw = self._client.call_tool(tool, **kwargs)
        return raw if isinstance(raw, str) else json.dumps(raw)


# ── Convenience constructor for Queue Doctor ───────────────────────────────

def QueueDoctorMCPAdapter(
    env_url: str,
    task_id: str = "task_1_easy",
    seed: int = 42,
) -> MCPAdapter:
    """
    Pre-configured MCPAdapter for the Queue Doctor environment.

    Args:
        env_url: Queue Doctor HF Space URL.
        task_id: "task_1_easy" | "task_2_medium" | "task_3_hard"
        seed:    Episode seed for stochasticity.

    Returns a fully configured MCPAdapter instance.
    """
    return MCPAdapter(
        env_url=env_url,
        init_tool="start_task",
        init_kwargs={"task_id": task_id},
        observe_tool="get_queue_state",
        finalize_tool="finalize_episode",
        done_key="done",
        reward_key="step_reward",
        state_key="state",
        score_key="score",
        action_map={
            "serve_patient": ("serve_patient", ["patient_id"]),
            "wait":          ("wait",          []),
        },
        default_action="wait",
    )


# ── JSON helpers ───────────────────────────────────────────────────────────

def _parse_json(raw: str | Any) -> dict:
    """Parse JSON string to dict. Returns {} on failure."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def extract_json_from_obs(text: str) -> dict:
    """
    Extract JSON object from potentially rephrased observation string.
    Handles all 5 ConsistencyProbe paraphrase templates.
    Public — can be used by agent functions in eval scripts.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}