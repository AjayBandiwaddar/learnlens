"""
learnlens/adapters/custom_adapter.py

CustomAdapter — base class for any custom RL environment.

This is the entry point for users who have built their own RL
environment outside of OpenEnv, Gymnasium, or SB3. If your
environment has reset(), step(), and returns a scalar reward,
it is compatible with LearnLens in under 20 lines.

Two ways to use this:

─────────────────────────────────────────────────────────────
WAY 1: Subclass (recommended for complex environments)
─────────────────────────────────────────────────────────────

    from learnlens.adapters.custom_adapter import CustomAdapter

    class MyTradingEnvAdapter(CustomAdapter):

        def __init__(self, data_path: str):
            self._env = TradingEnvironment(data=data_path)

        def reset(self, seed=None) -> StepResult:
            obs = self._env.reset(seed=seed)
            return self._make_result(obs, 0.0, False)

        def step(self, action: dict) -> StepResult:
            price_delta = action.get('delta', 0.0)
            obs, reward, done = self._env.step(price_delta)
            return self._make_result(obs, reward, done)

        def get_state(self) -> dict:
            return self._env.info()

─────────────────────────────────────────────────────────────
WAY 2: Wrap inline with from_callable() (quick, no subclass)
─────────────────────────────────────────────────────────────

    adapter = CustomAdapter.from_callable(
        reset_fn=lambda seed: my_env.reset(seed),
        step_fn=lambda action: my_env.step(action['move']),
        state_fn=lambda: my_env.info(),
        env_name="MyEnv-v1",
    )

    wrapper = LensWrapper(adapter=adapter)
    results = wrapper.evaluate(agent_fn=my_agent)

─────────────────────────────────────────────────────────────
WAY 3: Pass any object with reset()/step() methods
─────────────────────────────────────────────────────────────

    adapter = CustomAdapter.from_object(
        env_obj=my_env,
        action_key='move',   # which key from agent_fn dict to pass
        env_name="MyEnv-v1",
    )

Interface contract for subclasses:
  - reset(seed=None) -> StepResult        REQUIRED
  - step(action: dict) -> StepResult      REQUIRED
  - get_state() -> dict                   OPTIONAL (default: {})
  - open() -> None                        OPTIONAL (default: no-op)
  - close() -> None                       OPTIONAL (default: no-op)
  - health_check() -> bool                OPTIONAL (default: True)

Helper available in all subclasses:
  - self._make_result(obs, reward, done, metadata={}) -> StepResult
  - self.observation_to_str(obs) -> str    (static, matches OpenEnvAdapter)
"""

from __future__ import annotations

import json
from typing import Any, Callable

from learnlens.adapters.openenv import StepResult


class CustomAdapter:
    """
    Base class for custom RL environment adapters.

    Subclass this to connect any environment to LearnLens.
    Implement reset() and step() at minimum. Everything else
    has a working default.

    Attributes
    ----------
    env_name : str
        Human-readable name for this environment. Used in reports,
        logs, and the env_url property.
    """

    def __init__(self, env_name: str = "CustomEnv") -> None:
        self.env_name = env_name

    # ── Lifecycle — override if your env needs setup/teardown ─────────────────

    def open(self) -> None:
        """
        Optional setup. Called before the first reset() or step().
        Override if your environment needs initialisation (e.g. opening
        a file, connecting to a server, loading a model).
        """
        pass

    def close(self) -> None:
        """
        Optional teardown. Called when evaluation is complete.
        Override if your environment holds resources (file handles,
        network connections, GPU memory).
        """
        pass

    def __enter__(self) -> "CustomAdapter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Returns True if the environment is ready to use.
        Default: always True (local envs are always available).
        Override if your env has an external dependency to check.
        """
        return True

    # ── Core API — override reset() and step() ────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """
        Reset the environment to an initial state.

        MUST be overridden in subclasses.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        StepResult
            Use self._make_result(obs, 0.0, False) to construct.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.reset() is not implemented.\n"
            "Override reset() in your subclass:\n\n"
            "    def reset(self, seed=None):\n"
            "        obs = self._env.reset(seed=seed)\n"
            "        return self._make_result(obs, 0.0, False)\n"
        )

    def step(self, action: dict) -> StepResult:
        """
        Take one step in the environment.

        MUST be overridden in subclasses.

        Parameters
        ----------
        action : dict
            Action dict from agent_fn. Extract the value you need:
            action.get('action') or action.get('move') etc.

        Returns
        -------
        StepResult
            Use self._make_result(obs, reward, done) to construct.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.step() is not implemented.\n"
            "Override step() in your subclass:\n\n"
            "    def step(self, action: dict):\n"
            "        raw = action.get('action')\n"
            "        obs, reward, done = self._env.step(raw)\n"
            "        return self._make_result(obs, reward, done)\n"
        )

    def get_state(self) -> dict:
        """
        Return metadata about the current environment state.
        Default: returns env_name only. Override to add more context.
        """
        return {"env_name": self.env_name}

    # ── Helper: construct StepResult ──────────────────────────────────────────

    def _make_result(
        self,
        obs: Any,
        reward: float,
        done: bool,
        metadata: dict | None = None,
    ) -> StepResult:
        """
        Construct a StepResult from raw environment outputs.

        Use this in your reset() and step() implementations instead
        of constructing StepResult directly — it handles observation
        normalisation and type safety automatically.

        Parameters
        ----------
        obs : any
            Raw observation from your environment. Can be:
              - dict         → passed through as-is
              - list/tuple   → {"observation": list}
              - numpy array  → {"observation": list, "shape": list, "dtype": str}
              - str          → {"observation": str}
              - scalar       → {"observation": value}
        reward : float
            Scalar reward signal.
        done : bool
            Whether the episode has ended.
        metadata : dict | None
            Optional additional metadata to include in StepResult.

        Returns
        -------
        StepResult
        """
        return StepResult(
            observation=self._format_obs(obs),
            reward=float(reward),
            done=bool(done),
            metadata={
                "env_name": self.env_name,
                **(metadata or {}),
            },
        )

    # ── Observation formatting ────────────────────────────────────────────────

    @staticmethod
    def _format_obs(obs: Any) -> dict:
        """
        Normalise any observation type to a dict for StepResult.

        Handles numpy arrays, dicts, lists, scalars, and strings.
        Consistent with GymnasiumAdapter's observation_formatter.
        """
        try:
            import numpy as np
            if isinstance(obs, np.ndarray):
                return {
                    "observation": obs.tolist(),
                    "shape": list(obs.shape),
                    "dtype": str(obs.dtype),
                }
        except ImportError:
            pass

        if isinstance(obs, dict):
            return obs
        if isinstance(obs, (list, tuple)):
            return {"observation": list(obs)}
        if isinstance(obs, str):
            return {"observation": obs}
        return {"observation": obs}

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        """
        Serialise observation to string for agent_fn.
        Matches OpenEnvAdapter API — all LearnLens probes use this.
        """
        if isinstance(obs, str):
            return obs
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)

    # ── Compatibility property ────────────────────────────────────────────────

    @property
    def env_url(self) -> str:
        """Compatibility property — returns env identifier string."""
        return f"custom://{self.env_name}"

    # ── Class method constructors ─────────────────────────────────────────────

    @classmethod
    def from_callable(
        cls,
        reset_fn: Callable,
        step_fn: Callable,
        state_fn: Callable | None = None,
        env_name: str = "CustomEnv",
    ) -> "CustomAdapter":
        """
        Create a CustomAdapter from callable functions.

        No subclassing required. Pass your reset and step functions
        directly. This is the fastest way to wrap a custom environment.

        Parameters
        ----------
        reset_fn : Callable
            Function: (seed: int | None) -> (obs, optional_info)
            or just: (seed) -> obs
        step_fn : Callable
            Function: (action_value) -> (obs, reward, done)
            or: (action_value) -> (obs, reward, terminated, truncated, info)
            Receives the raw action value (not the dict).
        state_fn : Callable | None
            Optional function: () -> dict
        env_name : str
            Human-readable name for logging and reports.

        Example
        -------
            adapter = CustomAdapter.from_callable(
                reset_fn=lambda seed: my_env.reset(seed),
                step_fn=lambda action: my_env.step(action),
                env_name="MyEnv-v1",
            )
        """
        adapter = _CallableAdapter(
            reset_fn=reset_fn,
            step_fn=step_fn,
            state_fn=state_fn,
            env_name=env_name,
        )
        return adapter

    @classmethod
    def from_object(
        cls,
        env_obj: Any,
        action_key: str = "action",
        env_name: str | None = None,
        seed_kwarg: str = "seed",
    ) -> "CustomAdapter":
        """
        Create a CustomAdapter from any object with reset() and step().

        Works with any object that has:
          - env_obj.reset(seed=None) -> obs or (obs, info)
          - env_obj.step(action) -> (obs, reward, done) or 5-tuple

        Parameters
        ----------
        env_obj : Any
            The environment object to wrap.
        action_key : str
            Which key to extract from the agent_fn action dict.
            Default: "action". Change to "move", "values", etc.
            to match your agent_fn's output.
        env_name : str | None
            Human-readable name. Defaults to type(env_obj).__name__.
        seed_kwarg : str
            Keyword argument name for seed in env_obj.reset().
            Default: "seed". Change if your env uses "random_seed" etc.

        Example
        -------
            adapter = CustomAdapter.from_object(
                env_obj=my_chess_env,
                action_key='move',
                env_name='ChessEnv-v1',
            )
        """
        name = env_name or type(env_obj).__name__

        def _reset_fn(seed=None):
            try:
                return env_obj.reset(**{seed_kwarg: seed})
            except TypeError:
                return env_obj.reset()

        def _step_fn(action_value):
            return env_obj.step(action_value)

        def _state_fn():
            try:
                return env_obj.state() or {}
            except AttributeError:
                return {}

        return _CallableAdapter(
            reset_fn=_reset_fn,
            step_fn=_step_fn,
            state_fn=_state_fn,
            env_name=name,
            action_key=action_key,
        )


# ── Private: callable-based concrete implementation ───────────────────────────

class _CallableAdapter(CustomAdapter):
    """
    Concrete implementation of CustomAdapter backed by callables.
    Not part of the public API — use CustomAdapter.from_callable()
    or CustomAdapter.from_object() instead.
    """

    def __init__(
        self,
        reset_fn: Callable,
        step_fn: Callable,
        state_fn: Callable | None,
        env_name: str,
        action_key: str = "action",
    ) -> None:
        super().__init__(env_name=env_name)
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._state_fn = state_fn
        self._action_key = action_key

    def reset(self, seed: int | None = None) -> StepResult:
        raw = self._reset_fn(seed)
        # Handle (obs, info) tuple vs bare obs
        if isinstance(raw, tuple) and len(raw) == 2:
            obs, _ = raw
        else:
            obs = raw
        return self._make_result(obs, 0.0, False)

    def step(self, action: dict) -> StepResult:
        # Extract action value from dict
        raw_action = (
            action.get(self._action_key)
            or action.get("action")
            or action.get("values")
            or action.get("value")
            or next(iter(action.values()), None)
        )
        raw = self._step_fn(raw_action)

        # Normalise return format
        if isinstance(raw, tuple):
            if len(raw) == 5:
                obs, reward, terminated, truncated, info = raw
                done = terminated or truncated
            elif len(raw) == 4:
                obs, reward, done, _ = raw
            elif len(raw) == 3:
                obs, reward, done = raw
            else:
                obs, reward, done = raw[0], 0.0, False
        else:
            obs, reward, done = raw, 0.0, False

        return self._make_result(obs, float(reward), bool(done))

    def get_state(self) -> dict:
        if self._state_fn is not None:
            try:
                return self._state_fn() or {"env_name": self.env_name}
            except Exception:
                pass
        return {"env_name": self.env_name}