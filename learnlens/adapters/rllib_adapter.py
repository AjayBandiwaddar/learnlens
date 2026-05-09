"""
learnlens/adapters/rllib_adapter.py

RLlibAdapter — wraps trained Ray RLlib algorithms for LearnLens evaluation.

Covers every RLlib algorithm without modification:
  On-policy  : PPO, A2C, IMPALA, APPO
  Off-policy : SAC, TD3, DDPG, DQN, R2D2
  Model-based: DreamerV3, MuZero
  Multi-agent: via policy_id parameter

RLlib is the dominant production RL framework — used at scale by
Uber, Lyft, Microsoft, and most large ML teams running distributed RL.
This adapter makes those trained policies immediately diagnosable with LQS.

Two usage patterns:

─────────────────────────────────────────────────────────────────────
PATTERN 1: Trained RLlib Algorithm + environment id (most common)
─────────────────────────────────────────────────────────────────────

    import ray
    from ray.rllib.algorithms.ppo import PPO

    ray.init()
    algo = PPO.from_checkpoint("/path/to/checkpoint")

    from learnlens.adapters.rllib_adapter import RLlibAdapter
    from learnlens.wrapper import LensWrapper

    adapter = RLlibAdapter(algo=algo, env_id="CartPole-v1")
    wrapper = LensWrapper(adapter=adapter)
    results = wrapper.evaluate(agent_fn=adapter.as_agent_fn())
    results.print_report()
    print(results.lqs)

─────────────────────────────────────────────────────────────────────
PATTERN 2: Custom registered RLlib environment
─────────────────────────────────────────────────────────────────────

    import ray
    from ray import tune
    from ray.rllib.algorithms.sac import SAC

    ray.init()
    tune.register_env("MyEnv-v1", lambda cfg: MyCustomEnv(cfg))
    algo = SAC.from_checkpoint("/path/to/checkpoint")

    adapter = RLlibAdapter(
        algo=algo,
        env_id="MyEnv-v1",
        env_config={"difficulty": "hard"},
    )

─────────────────────────────────────────────────────────────────────
PATTERN 3: Multi-agent policy evaluation
─────────────────────────────────────────────────────────────────────

    adapter = RLlibAdapter(
        algo=algo,
        env_id="MultiAgentEnv-v1",
        policy_id="agent_0",        # which policy to evaluate
    )

─────────────────────────────────────────────────────────────────────
PATTERN 4: Environment-only (no trained model — use custom agent_fn)
─────────────────────────────────────────────────────────────────────

    adapter = RLlibAdapter(env_id="CartPole-v1")
    wrapper = LensWrapper(adapter=adapter)
    results = wrapper.evaluate(agent_fn=my_custom_agent)

Notes on Ray:
    RLlib requires Ray to be initialised before use: ray.init()
    The adapter checks for this and raises a clear error if Ray is not running.
    Ray does NOT need to be initialised just to use the environment interface
    (reset/step) — only as_agent_fn() requires a trained algo.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from learnlens.adapters.openenv import StepResult
from learnlens.adapters.custom_adapter import CustomAdapter


class RLlibAdapter(CustomAdapter):
    """
    Wraps a trained Ray RLlib algorithm for LearnLens evaluation.

    Inherits from CustomAdapter — all probe infrastructure works
    identically. Extends with RLlib-specific algorithm handling,
    checkpoint loading, and multi-agent policy support.

    Parameters
    ----------
    algo : ray.rllib.algorithms.Algorithm | None
        A trained RLlib algorithm. Load with:
            PPO.from_checkpoint("/path/to/checkpoint")
        Pass None if you only want the environment interface
        and will provide your own agent_fn.
    env_id : str | None
        Gymnasium or registered RLlib environment ID.
        Required if env is not provided.
    env : Any | None
        Pre-instantiated environment object. Takes precedence over env_id.
        Must implement reset() and step().
    env_config : dict | None
        Configuration dict passed to the environment constructor.
        Used when env_id refers to a registered RLlib environment.
    policy_id : str
        Which policy to use for action prediction in multi-agent settings.
        Default: "default_policy" (correct for single-agent algorithms).
    deterministic : bool
        Whether to use deterministic action prediction.
        True = exploit mode (evaluation). False = explore mode (testing robustness).
    explore : bool | None
        Explicit explore flag passed to compute_single_action().
        If None, derived from deterministic parameter.
    """

    def __init__(
        self,
        algo: Any | None = None,
        env_id: str | None = None,
        env: Any | None = None,
        env_config: dict | None = None,
        policy_id: str = "default_policy",
        deterministic: bool = True,
        explore: bool | None = None,
    ) -> None:
        if env_id is None and env is None:
            raise ValueError(
                "Either env_id (string) or env (environment object) must be provided."
            )

        env_name = env_id or type(env).__name__
        super().__init__(env_name=env_name)

        self.algo = algo
        self._env_id = env_id
        self._env_config = env_config or {}
        self._policy_id = policy_id
        self._deterministic = deterministic
        self._explore = explore if explore is not None else (not deterministic)

        # Environment — either provided or built from env_id
        self._raw_env: Any = env
        self._env_built = env is not None  # True if we didn't create it

        # Cache for last raw numpy obs (for compute_single_action)
        self._last_raw_obs: np.ndarray | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Build the environment if not already provided."""
        if self._raw_env is not None:
            return
        self._raw_env = self._build_env()

    def close(self) -> None:
        """Close the environment if this adapter created it."""
        if self._raw_env is not None and not self._env_built:
            try:
                self._raw_env.close()
            except Exception:
                pass
        self._raw_env = None

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Returns True if the environment is available.
        Does NOT require Ray to be initialised — env check only.
        """
        try:
            self._ensure_env()
            return self._raw_env is not None
        except Exception:
            return False

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """
        Reset the RLlib environment.

        Parameters
        ----------
        seed : int | None
            Random seed. Passed to env.reset() if supported.

        Returns
        -------
        StepResult with rllib_env metadata.
        """
        self._ensure_env()

        try:
            # RLlib envs follow Gymnasium API (returns obs, info)
            result = self._raw_env.reset(seed=seed)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs, info = result, {}
        except TypeError:
            # Older RLlib env — reset() takes no args
            obs = self._raw_env.reset()
            info = {}

        self._last_raw_obs = self._to_numpy(obs)

        return self._make_result(
            obs=obs,
            reward=0.0,
            done=False,
            metadata={
                "info": info if isinstance(info, dict) else {},
                "env_id": self._env_id,
                "policy_id": self._policy_id,
            },
        )

    def step(self, action: dict) -> StepResult:
        """
        Take one step in the RLlib environment.

        Parameters
        ----------
        action : dict
            Action dict from agent_fn: {"action": <value>}

        Returns
        -------
        StepResult with rllib_env metadata.
        """
        self._ensure_env()

        raw_action = self._extract_action(action)

        try:
            result = self._raw_env.step(raw_action)
        except Exception as exc:
            raise RuntimeError(
                f"RLlib env step() failed for {self._env_id}: {exc}\n"
                f"Action received: {action}"
            ) from exc

        # Handle both 4-tuple and 5-tuple step returns
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, info = result[0], 0.0, False, {}

        self._last_raw_obs = self._to_numpy(obs)

        return self._make_result(
            obs=obs,
            reward=float(reward),
            done=bool(done),
            metadata={
                "info": info if isinstance(info, dict) else {},
                "env_id": self._env_id,
                "policy_id": self._policy_id,
            },
        )

    def get_state(self) -> dict:
        """Return environment and algorithm metadata."""
        state = {
            "env_name": self.env_name,
            "env_id": self._env_id,
            "policy_id": self._policy_id,
            "deterministic": self._deterministic,
        }
        if self.algo is not None:
            state.update(self._algo_metadata())
        return state

    # ── The key method: as_agent_fn() ─────────────────────────────────────────

    def as_agent_fn(self) -> Callable[[str], dict]:
        """
        Convert the trained RLlib algorithm into a LearnLens agent_fn.

        Uses algo.compute_single_action() internally — the standard
        RLlib inference API for single-environment evaluation.

        Returns
        -------
        Callable[[str], dict]
            A function: agent_fn(obs_str: str) -> action_dict

        Raises
        ------
        RuntimeError
            If no algo was provided at construction time.
        ValueError
            If Ray is not initialised (ray.init() not called).

        Example
        -------
            wrapper = LensWrapper(adapter=adapter)
            results = wrapper.evaluate(agent_fn=adapter.as_agent_fn())
        """
        if self.algo is None:
            raise RuntimeError(
                "as_agent_fn() requires a trained RLlib algorithm.\n"
                "Pass algo=PPO.from_checkpoint('/path') to the adapter,\n"
                "or provide your own agent_fn to wrapper.evaluate()."
            )

        self._check_ray_init()

        algo = self.algo
        policy_id = self._policy_id
        explore = self._explore
        adapter_ref = self

        def _agent_fn(obs_str: str) -> dict:
            """
            LearnLens agent_fn wrapping algo.compute_single_action().

            Observation flow:
              obs_str (JSON string from observation_to_str())
              → parse to numpy array
              → algo.compute_single_action(obs, policy_id=policy_id)
              → convert action to LearnLens dict format
            """
            raw_obs = adapter_ref._parse_obs_str(obs_str)

            # compute_single_action is RLlib's standard single-env inference API
            action = algo.compute_single_action(
                observation=raw_obs,
                policy_id=policy_id,
                explore=explore,
            )

            return adapter_ref._action_to_dict(action)

        return _agent_fn

    # ── Checkpoint loading ────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        algo_class: Any | str,
        env_id: str,
        env_config: dict | None = None,
        policy_id: str = "default_policy",
        deterministic: bool = True,
    ) -> "RLlibAdapter":
        """
        Load a trained RLlib algorithm from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the RLlib checkpoint directory.
        algo_class : type | str
            RLlib algorithm class or string name.
            Accepted strings: "PPO", "SAC", "DQN", "TD3", "A2C",
            "DDPG", "IMPALA", "APPO".
            Or pass the class directly: from ray.rllib.algorithms.ppo import PPO
        env_id : str
            Gymnasium or registered RLlib environment ID.
        env_config : dict | None
            Environment configuration.
        policy_id : str
            Policy to evaluate in multi-agent settings.
        deterministic : bool
            Whether to use deterministic prediction.

        Example
        -------
            adapter = RLlibAdapter.from_checkpoint(
                checkpoint_path="/path/to/checkpoint",
                algo_class="PPO",
                env_id="CartPole-v1",
            )
        """
        resolved_class = cls._resolve_algo_class(algo_class)
        algo = resolved_class.from_checkpoint(checkpoint_path)
        return cls(
            algo=algo,
            env_id=env_id,
            env_config=env_config,
            policy_id=policy_id,
            deterministic=deterministic,
        )

    # ── Model management ──────────────────────────────────────────────────────

    def swap_algo(self, new_algo: Any) -> None:
        """
        Swap the trained algorithm while keeping the same environment.

        Allows comparing multiple checkpoints on the same environment
        without re-instantiating the adapter.

        Parameters
        ----------
        new_algo : ray.rllib.algorithms.Algorithm
            A trained RLlib algorithm to replace the current one.
        """
        if not hasattr(new_algo, "compute_single_action"):
            raise ValueError(
                "new_algo must have compute_single_action(). "
                "Expected a trained RLlib Algorithm."
            )
        self.algo = new_algo

    # ── Compatibility properties ──────────────────────────────────────────────

    @property
    def env_url(self) -> str:
        algo_name = type(self.algo).__name__ if self.algo else "NoAlgo"
        return f"rllib://{algo_name}/{self._env_id or self.env_name}"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_env(self) -> None:
        if self._raw_env is None:
            self.open()

    def _build_env(self) -> Any:
        """
        Build the environment from env_id.

        Tries Gymnasium first (most common). If the env_id is not
        registered in Gymnasium, tries RLlib's environment registry.
        """
        # Try Gymnasium first
        try:
            import gymnasium as gym
            env = gym.make(self._env_id, **self._env_config)
            return env
        except Exception:
            pass

        # Try RLlib environment registry
        try:
            from ray.tune.registry import ENV_CREATOR, _global_registry
            if _global_registry.contains(ENV_CREATOR, self._env_id):
                creator = _global_registry.get(ENV_CREATOR, self._env_id)
                return creator(self._env_config)
        except Exception:
            pass

        raise RuntimeError(
            f"Cannot build environment '{self._env_id}'.\n"
            "Ensure either:\n"
            "  1. The env_id is a valid Gymnasium environment, or\n"
            "  2. The environment is registered with:\n"
            "     ray.tune.register_env('MyEnv', lambda cfg: MyEnv(cfg))\n"
            "  3. Pass env=my_env_object directly to avoid auto-building."
        )

    def _algo_metadata(self) -> dict:
        """Extract algorithm metadata for get_state() and StepResult."""
        meta = {
            "algo": type(self.algo).__name__,
            "policy_id": self._policy_id,
            "deterministic": self._deterministic,
        }
        # num_env_steps_trained if available
        if hasattr(self.algo, "training_iteration"):
            meta["training_iteration"] = self.algo.training_iteration
        return meta

    @staticmethod
    def _check_ray_init() -> None:
        """Verify Ray is initialised before using algo inference."""
        try:
            import ray
            if not ray.is_initialized():
                raise ValueError(
                    "Ray is not initialised. Call ray.init() before using "
                    "RLlibAdapter.as_agent_fn().\n\n"
                    "    import ray\n"
                    "    ray.init()\n"
                )
        except ImportError as exc:
            raise ImportError(
                "ray[rllib] is required: pip install 'ray[rllib]'"
            ) from exc

    @staticmethod
    def _resolve_algo_class(algo_class: Any | str) -> Any:
        """Resolve algorithm class from string name or return directly."""
        if not isinstance(algo_class, str):
            return algo_class

        name = algo_class.upper()
        try:
            if name == "PPO":
                from ray.rllib.algorithms.ppo import PPO
                return PPO
            elif name == "SAC":
                from ray.rllib.algorithms.sac import SAC
                return SAC
            elif name == "DQN":
                from ray.rllib.algorithms.dqn import DQN
                return DQN
            elif name == "TD3":
                from ray.rllib.algorithms.td3 import TD3
                return TD3
            elif name == "A2C":
                from ray.rllib.algorithms.a2c import A2C
                return A2C
            elif name == "DDPG":
                from ray.rllib.algorithms.ddpg import DDPG
                return DDPG
            elif name == "IMPALA":
                from ray.rllib.algorithms.impala import IMPALA
                return IMPALA
            elif name == "APPO":
                from ray.rllib.algorithms.appo import APPO
                return APPO
            else:
                raise ValueError(
                    f"Unknown RLlib algorithm: {algo_class}.\n"
                    "Supported: PPO, SAC, DQN, TD3, A2C, DDPG, IMPALA, APPO.\n"
                    "Or pass the class directly: algo_class=PPO"
                )
        except ImportError as exc:
            raise ImportError(
                "ray[rllib] is required: pip install 'ray[rllib]'"
            ) from exc

    @staticmethod
    def _extract_action(action_dict: dict) -> Any:
        """Extract raw action value from LearnLens action dict."""
        raw = (
            action_dict.get("action")
            or action_dict.get("values")
            or action_dict.get("value")
            or action_dict.get("move")
        )
        if raw is None and action_dict:
            raw = next(iter(action_dict.values()), None)
        return raw

    @staticmethod
    def _to_numpy(obs: Any) -> np.ndarray | Any:
        """Convert observation to numpy array for compute_single_action."""
        if isinstance(obs, np.ndarray):
            return obs
        if isinstance(obs, (list, tuple)):
            return np.array(obs, dtype=np.float32)
        if isinstance(obs, dict):
            # Dict observation space — return as-is (RLlib handles it)
            return obs
        return obs

    @staticmethod
    def _parse_obs_str(obs_str: str) -> np.ndarray | dict:
        """Parse observation string back to numpy for compute_single_action."""
        if isinstance(obs_str, np.ndarray):
            return obs_str
        try:
            obs_dict = json.loads(obs_str)
            if isinstance(obs_dict, dict):
                raw = obs_dict.get("observation")
                if raw is not None:
                    dtype_str = obs_dict.get("dtype", "float32")
                    shape = obs_dict.get("shape")
                    arr = np.array(raw, dtype=dtype_str)
                    if shape:
                        arr = arr.reshape(shape)
                    return arr
                # Dict observation space — return as-is
                return obs_dict
            return np.array(obs_dict, dtype=np.float32)
        except (json.JSONDecodeError, TypeError):
            try:
                return np.array(eval(obs_str))  # noqa: S307
            except Exception:
                raise ValueError(
                    f"Cannot parse observation string: {obs_str[:100]!r}"
                )

    @staticmethod
    def _action_to_dict(action: Any) -> dict:
        """Convert RLlib action output to LearnLens action dict."""
        if isinstance(action, np.ndarray):
            return {"action": action.tolist()}
        if isinstance(action, (np.integer, np.floating)):
            return {"action": action.item()}
        if isinstance(action, dict):
            return action  # Multi-agent action dict — pass through
        return {"action": action}