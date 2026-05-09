"""
learnlens/adapters/sb3_adapter.py

StableBaselines3Adapter — wraps trained SB3 models for LearnLens evaluation.

Bridges the gap between:
  - SB3 models (predict on numpy arrays, return numpy actions)
  - LearnLens probes (agent_fn takes str observation, returns action dict)

Supported algorithms (anything in stable-baselines3):
  On-policy  : PPO, A2C
  Off-policy : SAC, TD3, DDPG, DQN
  Goal-based : HER (with any of the above)

Usage — three patterns:

  Pattern 1: evaluate a trained model directly
  ------------------------------------------
  from stable_baselines3 import PPO
  from learnlens.adapters.sb3_adapter import StableBaselines3Adapter
  from learnlens.wrapper import LensWrapper

  model = PPO.load("cartpole_ppo")
  adapter = StableBaselines3Adapter(model, "CartPole-v1")
  wrapper = LensWrapper(adapter=adapter)
  results = wrapper.evaluate(agent_fn=adapter.as_agent_fn())
  print(results.lqs)

  Pattern 2: compare multiple trained models on the same env
  ----------------------------------------------------------
  for name, model in {"ppo": ppo_model, "a2c": a2c_model}.items():
      adapter = StableBaselines3Adapter(model, "CartPole-v1")
      results = LensWrapper(adapter=adapter).evaluate(
          agent_fn=adapter.as_agent_fn()
      )
      print(f'{name} LQS: {results.lqs:.3f}')

  Pattern 3: stochastic policy evaluation (for ConsistencyProbe)
  -------------------------------------------------------------
  adapter = StableBaselines3Adapter(model, "LunarLander-v2",
                                    deterministic=False)

Design notes:
  - reset() and step() delegate to an internal GymnasiumAdapter
  - Raw numpy observation is cached internally after each reset()/step()
  - as_agent_fn() returns a closure that calls model.predict() on the
    cached numpy obs — bypasses the lossy string serialisation round-trip
  - VecEnv reshape is handled automatically
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from learnlens.adapters.openenv import StepResult
from learnlens.adapters.gymnasium_adapter import GymnasiumAdapter


class StableBaselines3Adapter:
    """
    Wraps a trained Stable-Baselines3 model and evaluation environment.

    Parameters
    ----------
    model : stable_baselines3 base algorithm
        Any trained SB3 model (PPO, SAC, DQN, A2C, TD3, DDPG).
        Must expose model.predict(obs, deterministic) -> (action, states).
    env_or_id : str | gymnasium.Env | GymnasiumAdapter
        Evaluation environment. Should be a FRESH env, not the training one,
        to avoid data leakage across evaluation probes.
    deterministic : bool
        Whether model.predict() uses the deterministic policy.
        Default True — consistent actions for the same observation.
        Set False to sample the stochastic policy.
    render_mode : str | None
        Passed to GymnasiumAdapter when env_or_id is a string.
    """

    def __init__(
        self,
        model: Any,
        env_or_id: "str | Any",
        deterministic: bool = True,
        render_mode: str | None = None,
    ) -> None:
        self._model = model
        self._deterministic = deterministic

        if isinstance(env_or_id, GymnasiumAdapter):
            self._gym_adapter = env_or_id
            self._owns_adapter = False
        else:
            self._gym_adapter = GymnasiumAdapter(
                env_or_id, render_mode=render_mode
            )
            self._owns_adapter = True

        # Raw numpy obs cache — updated on every reset()/step()
        # model.predict() needs the exact numpy array, not the JSON form
        self._last_raw_obs: np.ndarray | None = None
        self._needs_vec_reshape: bool = self._detect_vec_model()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        self._gym_adapter.open()

    def close(self) -> None:
        if self._owns_adapter:
            self._gym_adapter.close()

    def __enter__(self) -> "StableBaselines3Adapter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            return (
                self._model is not None
                and self._gym_adapter.health_check()
            )
        except Exception:
            return False

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> StepResult:
        """
        Reset the environment and cache the raw numpy observation.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility across evaluation episodes.

        Returns
        -------
        StepResult
            observation: formatted dict, reward: 0.0, done: False
        """
        result = self._gym_adapter.reset(seed=seed)
        self._last_raw_obs = self._extract_numpy_obs(result.observation)
        return result

    def step(self, action: dict) -> StepResult:
        """
        Take one step. Caches the resulting raw numpy observation.

        Parameters
        ----------
        action : dict
            Action dict — typically produced by as_agent_fn() but can
            also be constructed manually for probe testing.

        Returns
        -------
        StepResult
            next observation, reward, done flag, metadata
        """
        result = self._gym_adapter.step(action)
        self._last_raw_obs = self._extract_numpy_obs(result.observation)
        return result

    def get_state(self) -> dict:
        """
        Return environment and model metadata.
        Extends GymnasiumAdapter.get_state() with model information.
        """
        state = self._gym_adapter.get_state()
        state.update({
            "model_class": type(self._model).__name__,
            "deterministic": self._deterministic,
            "needs_vec_reshape": self._needs_vec_reshape,
            "policy_class": type(
                getattr(self._model, "policy", None)
            ).__name__,
        })
        return state

    # ── The key method ────────────────────────────────────────────────────────

    def as_agent_fn(self) -> Callable[[str], dict]:
        """
        Returns a LearnLens-compatible agent_fn closure.

        The returned callable:
          1. Ignores the string observation argument (uses the cached
             raw numpy obs from the last reset() or step() instead —
             avoids the lossy JSON string round-trip)
          2. Calls model.predict(obs, deterministic=self._deterministic)
          3. Converts the numpy action to a LearnLens action dict

        Usage:
            adapter = StableBaselines3Adapter(model, "CartPole-v1")
            wrapper = LensWrapper(adapter=adapter)
            results = wrapper.evaluate(agent_fn=adapter.as_agent_fn())

        Returns
        -------
        Callable[[str], dict]
            Compatible with LensWrapper.evaluate(agent_fn=...)
        """
        adapter = self

        def agent_fn(observation_str: str) -> dict:
            if adapter._last_raw_obs is None:
                raise RuntimeError(
                    "StableBaselines3Adapter: no cached observation. "
                    "Call adapter.reset() before agent_fn."
                )

            obs = adapter._last_raw_obs
            if adapter._needs_vec_reshape and obs.ndim > 0:
                obs = obs.reshape(1, *obs.shape)

            try:
                action, _states = adapter._model.predict(
                    obs, deterministic=adapter._deterministic
                )
            except Exception as exc:
                raise RuntimeError(
                    f"model.predict() failed: {exc}\n"
                    f"Obs shape: {obs.shape}, dtype: {obs.dtype}"
                ) from exc

            return adapter._numpy_action_to_dict(action)

        return agent_fn

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self) -> Any:
        """Direct access to the underlying SB3 model."""
        return self._model

    @property
    def env_url(self) -> str:
        env_id = getattr(self._gym_adapter, "_env_id", None) or "custom"
        return f"sb3://{type(self._model).__name__}/{env_id}"

    @staticmethod
    def observation_to_str(obs: dict | str) -> str:
        """Serialise observation to string. Matches OpenEnvAdapter API."""
        if isinstance(obs, str):
            return obs
        try:
            return json.dumps(obs, indent=2)
        except (TypeError, ValueError):
            return str(obs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_vec_model(self) -> bool:
        """
        Detect whether the SB3 model was trained on a VecEnv.
        VecEnv models expect obs shaped (n_envs, *obs_shape).
        """
        try:
            return getattr(self._model, "n_envs", 1) > 1
        except Exception:
            return False

    @staticmethod
    def _extract_numpy_obs(obs_dict: dict) -> np.ndarray:
        """
        Reconstruct raw numpy array from GymnasiumAdapter observation dict.
        Reverses _default_observation_formatter() exactly.

        {"observation": [...], "shape": [...], "dtype": "float32"}
            → np.array reshaped to shape with correct dtype
        {"observation": [...]}
            → np.float32 array
        dict obs space
            → concatenated flat float32 array
        """
        if "observation" in obs_dict and "shape" in obs_dict:
            dtype = obs_dict.get("dtype", "float32")
            return np.array(
                obs_dict["observation"], dtype=dtype
            ).reshape(obs_dict["shape"])

        if "observation" in obs_dict:
            val = obs_dict["observation"]
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=np.float32)
            return np.array([val], dtype=np.float32)

        # Dict observation space — stack values
        arrays = []
        for v in obs_dict.values():
            if isinstance(v, (list, np.ndarray)):
                arrays.append(np.array(v, dtype=np.float32).flatten())
            elif isinstance(v, (int, float)):
                arrays.append(np.array([v], dtype=np.float32))
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

    @staticmethod
    def _numpy_action_to_dict(action: Any) -> dict:
        """
        Convert SB3 model.predict() numpy output → LearnLens action dict.

        np.ndarray 0-d  → {"action": int or float scalar}
        np.ndarray (1,) → {"action": float}
        np.ndarray (n,) → {"action": [float, ...]}
        np.integer      → {"action": int}
        np.floating     → {"action": float}
        other           → {"action": action}
        """
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                val = action.item()
                return {"action": int(val) if isinstance(val, (int, np.integer)) else float(val)}
            if action.ndim == 1 and action.size == 1:
                val = action.item()
                return {"action": int(val) if action.dtype in (np.int32, np.int64) else float(val)}
            return {"action": action.tolist()}
        if isinstance(action, (np.integer, int)):
            return {"action": int(action)}
        if isinstance(action, (np.floating, float)):
            return {"action": float(action)}
        return {"action": action}