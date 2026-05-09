"""
learnlens/adapters/gymnasium_adapter.py

GymnasiumAdapter — wraps any Gymnasium-compatible environment.

Covers the entire Gymnasium ecosystem without modification:
  - Classic control : CartPole-v1, LunarLander-v2, MountainCar-v0,
                      Acrobot-v1, Pendulum-v1
  - Box2D           : BipedalWalker-v3, CarRacing-v2
  - Atari           : ALE/Pong-v5, ALE/Breakout-v5, ALE/SpaceInvaders-v5
  - MuJoCo          : HalfCheetah-v4, Hopper-v4, Ant-v4, Humanoid-v4
  - Custom envs     : any env registered via gymnasium.register()

Interface is identical to OpenEnvAdapter and DirectAdapter —
all LearnLens probes work unchanged against Gymnasium environments.

Action convention (agent_fn → Gymnasium):
  Discrete space  : agent returns {"action": 0}   → int 0
  Continuous space: agent returns {"action": [.1]} → np.array([.1])
  Multi-discrete  : agent returns {"action": [1,0]}→ np.array([1, 0])

Observation convention (Gymnasium → StepResult):
  numpy array     : {"observation": [...], "shape": [...], "dtype": "..."}
  dict obs space  : passed through as-is
  other           : {"observation": str(obs)}

Usage:
    import gymnasium as gym
    from learnlens.adapters.gymnasium_adapter import GymnasiumAdapter
    from learnlens.wrapper import LensWrapper

    # Option 1 — pass an env id string (adapter creates and manages env)
    adapter = GymnasiumAdapter("CartPole-v1")

    # Option 2 — pass an already-instantiated env
    env = gym.make("LunarLander-v2", render_mode=None)
    adapter = GymnasiumAdapter(env)

    # Option 3 — use directly with LensWrapper
    wrapper = LensWrapper(adapter=GymnasiumAdapter("HalfCheetah-v4"))
    results = wrapper.evaluate(agent_fn=my_agent)
    print(results.lqs)
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from learnlens.adapters.openenv import StepResult


class GymnasiumAdapter:
    """
    Wraps any Gymnasium-compatible environment for use with LearnLens.

    Parameters
    ----------
    env_or_id : str | gymnasium.Env
        Either a Gymnasium environment ID string (e.g. "CartPole-v1")
        or an already-instantiated Gymnasium environment object.
    render_mode : str | None
        Passed to gymnasium.make() when env_or_id is a string.
        Default None (no rendering) for evaluation speed.
    max_episode_steps : int | None
        Override the default step limit. None uses the environment default.
    action_parser : Callable | None
        Custom function to convert agent dict → Gymnasium action.
        If None, the default parser handles Discrete, Box, and MultiDiscrete.
    observation_formatter : Callable | None
        Custom function to convert Gymnasium obs → dict for StepResult.
        If None, the default formatter handles numpy arrays and dicts.
    """

    def __init__(
        self,
        env_or_id: "str | Any",
        render_mode: str | None = None,
        max_episode_steps: int | None = None,
        action_parser: Callable | None = None,
        observation_formatter: Callable | None = None,
    ) -> None:
        self._env_id: str | None = None
        self._env: Any = None
        self._render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._action_parser = action_parser or self._default_action_parser
        self._observation_formatter = (
            observation_formatter or self._default_observation_formatter
        )
        self._owned = False  # True if we created the env and must close it

        if isinstance(env_or_id, str):
            self._env_id = env_or_id
            # Defer creation to open() so the adapter is lightweight to construct
        else:
            self._env = env_or_id
            self._owned = False

        # Cached action space type for fast dispatch in step()
        self._action_space_type: str | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Initialise the Gymnasium environment.
        Auto-called on first reset() or step() if not called explicitly.
        """
        if self._env is not None:
            return  # Already open

        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "gymnasium is required: pip install gymnasium\n"
                "For MuJoCo:  pip install gymnasium[mujoco]\n"
                "For Atari:   pip install gymnasium[atari] autorom[accept-rom-license]"
            ) from exc

        kwargs: dict = {"render_mode": self._render_mode}
        if self._max_episode_steps is not None:
            kwargs["max_episode_steps"] = self._max_episode_steps

        self._env = gym.make(self._env_id, **kwargs)
        self._owned = True
        self._cache_action_space_type()

    def close(self) -> None:
        """Close the environment. Only closes if this adapter owns the env."""
        if self._env is not None and self._owned:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None

    def __enter__(self) -> "GymnasiumAdapter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Returns True if the environment is available.
        For Gymnasium, this checks that the env is instantiated.
        """
        try:
            self._ensure_open()
            return self._env is not None
        except Exception:
            return False

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility. Passed directly to
            gymnasium env.reset(seed=seed).

        Returns
        -------
        StepResult
            observation: formatted gymnasium obs
            reward: 0.0 (no reward on reset)
            done: False
            metadata: {"info": info_dict, "env_id": env_id}
        """
        self._ensure_open()
        try:
            obs, info = self._env.reset(seed=seed)
        except TypeError:
            # Older Gymnasium API — reset() returns obs only
            obs = self._env.reset()
            info = {}

        return StepResult(
            observation=self._observation_formatter(obs),
            reward=0.0,
            done=False,
            metadata={
                "info": info if isinstance(info, dict) else {},
                "env_id": self._env_id or str(type(self._env).__name__),
                "seed": seed,
            },
        )

    def step(self, action: dict) -> StepResult:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : dict
            Action dict from agent_fn. Expected format:
              {"action": <value>}
            where <value> is int, float, list, or np.array depending on
            the environment's action space.

        Returns
        -------
        StepResult
            observation: formatted next state
            reward: scalar reward from environment
            done: True if episode has terminated or truncated
            metadata: {"info": info_dict, "terminated": bool, "truncated": bool}
        """
        self._ensure_open()
        gym_action = self._action_parser(action, self._env)

        try:
            result = self._env.step(gym_action)
        except Exception as exc:
            raise RuntimeError(
                f"step() failed in {self._env_id or type(self._env).__name__}: {exc}\n"
                f"Action received: {action}\n"
                f"Action space:    {self._env.action_space}"
            ) from exc

        # Gymnasium ≥ 0.26: returns (obs, reward, terminated, truncated, info)
        # Gymnasium < 0.26:  returns (obs, reward, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            raise RuntimeError(
                f"Unexpected step() return length {len(result)} from "
                f"{self._env_id or type(self._env).__name__}"
            )

        return StepResult(
            observation=self._observation_formatter(obs),
            reward=float(reward),
            done=bool(done),
            metadata={
                "info": info if isinstance(info, dict) else {},
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "env_id": self._env_id or str(type(self._env).__name__),
            },
        )

    def get_state(self) -> dict:
        """
        Return metadata about the current environment state.
        Gymnasium does not have a native state() API, so this returns
        environment configuration metadata instead.
        """
        self._ensure_open()
        try:
            state = {
                "env_id": self._env_id or str(type(self._env).__name__),
                "action_space": str(self._env.action_space),
                "observation_space": str(self._env.observation_space),
                "action_space_type": self._action_space_type,
            }
            # Include spec if available
            if hasattr(self._env, "spec") and self._env.spec is not None:
                state["max_episode_steps"] = getattr(
                    self._env.spec, "max_episode_steps", None
                )
            return state
        except Exception:
            return {}

    # ── Observation helpers ───────────────────────────────────────────────────

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        """Serialise observation to string for agent_fn. Matches OpenEnvAdapter API."""
        if isinstance(obs, str):
            return obs
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._env is None:
            self.open()
        if self._action_space_type is None:
            self._cache_action_space_type()

    def _cache_action_space_type(self) -> None:
        """Cache action space type string for fast dispatch in step()."""
        if self._env is None:
            return
        space = self._env.action_space
        space_name = type(space).__name__
        self._action_space_type = space_name  # "Discrete", "Box", "MultiDiscrete", etc.

    # ── Default action parser ─────────────────────────────────────────────────

    @staticmethod
    def _default_action_parser(action_dict: dict, env: Any) -> Any:
        """
        Convert agent_fn dict output → Gymnasium action.

        Handles:
          Discrete      : {"action": 0}       → 0 (int)
          Box           : {"action": [0.1]}   → np.array([0.1])
          MultiDiscrete : {"action": [1, 0]}  → np.array([1, 0])
          MultiBinary   : {"action": [1,0,1]} → np.array([1, 0, 1])
          Fallback      : returns raw value unchanged

        Also accepts {"action": value} or {"values": value} or
        a flat dict {"0": 1, "1": 0} for indexed discrete actions.
        """
        import gymnasium as gym

        # Extract raw value — try common keys first
        raw = (
            action_dict.get("action")
            or action_dict.get("values")
            or action_dict.get("value")
            or action_dict.get("move")
        )

        if raw is None and action_dict:
            # Fallback: take first value in dict
            raw = next(iter(action_dict.values()))

        if raw is None:
            raise ValueError(
                f"GymnasiumAdapter: cannot extract action from dict {action_dict}.\n"
                "Expected at least one of: 'action', 'values', 'value', 'move'."
            )

        space = env.action_space

        # Discrete: agent can pass int or string of int
        if isinstance(space, gym.spaces.Discrete):
            try:
                return int(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Discrete action space expects int. Got: {raw!r}"
                )

        # Box: continuous action — convert to numpy array of correct dtype
        if isinstance(space, gym.spaces.Box):
            arr = np.array(raw, dtype=space.dtype)
            if arr.shape != space.shape:
                arr = arr.reshape(space.shape)
            return np.clip(arr, space.low, space.high)

        # MultiDiscrete: array of discrete values
        if isinstance(space, gym.spaces.MultiDiscrete):
            return np.array(raw, dtype=np.int64)

        # MultiBinary: binary array
        if isinstance(space, gym.spaces.MultiBinary):
            return np.array(raw, dtype=np.int8)

        # Unknown space — pass raw through and hope for the best
        return raw

    # ── Default observation formatter ─────────────────────────────────────────

    @staticmethod
    def _default_observation_formatter(obs: Any) -> dict:
        """
        Convert Gymnasium observation → dict for StepResult.

        numpy array   : {"observation": list, "shape": list, "dtype": str}
        dict          : passed through as-is (Dict observation space)
        tuple         : {"observation": list}
        scalar/other  : {"observation": value}
        """
        if isinstance(obs, np.ndarray):
            return {
                "observation": obs.tolist(),
                "shape": list(obs.shape),
                "dtype": str(obs.dtype),
            }
        if isinstance(obs, dict):
            # Dict observation space — recursively convert any numpy values
            return {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in obs.items()
            }
        if isinstance(obs, (list, tuple)):
            return {"observation": list(obs)}
        # Scalar, int, float, string
        return {"observation": obs}

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def env(self) -> Any:
        """Direct access to the underlying Gymnasium environment."""
        self._ensure_open()
        return self._env

    @property
    def action_space(self) -> Any:
        """The Gymnasium action space of the wrapped environment."""
        self._ensure_open()
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        """The Gymnasium observation space of the wrapped environment."""
        self._ensure_open()
        return self._env.observation_space

    @property
    def env_url(self) -> str:
        """Compatibility property — returns env identifier string."""
        return f"gymnasium://{self._env_id or type(self._env).__name__}"