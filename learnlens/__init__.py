"""
learnlens — Universal evaluation layer for RL environments.

Measures WHAT an agent learned, not just HOW MUCH reward it accumulated.
Works with OpenEnv, Gymnasium, Stable Baselines 3, Ray RLlib,
and any custom environment.

Quick start (OpenEnv):
    from learnlens import LensWrapper
    env = LensWrapper(env_url="https://your-openenv-space.hf.space")
    results = env.evaluate(agent_fn=my_agent)
    print(results.lqs)

Quick start (Gymnasium):
    from learnlens.adapters import GymnasiumAdapter
    from learnlens import LensWrapper
    wrapper = LensWrapper(adapter=GymnasiumAdapter("CartPole-v1"))
    results = wrapper.evaluate(agent_fn=my_agent)
    print(results.lqs)

Quick start (Stable Baselines3):
    from stable_baselines3 import PPO
    from learnlens.adapters import StableBaselines3Adapter
    from learnlens import LensWrapper
    model = PPO.load("ppo_cartpole.zip")
    adapter = StableBaselines3Adapter(model=model, env_id="CartPole-v1")
    wrapper = LensWrapper(adapter=adapter)
    results = wrapper.evaluate(agent_fn=adapter.as_agent_fn())
    print(results.lqs)

pip install learnlens-rl
pip install learnlens-rl[gymnasium]   # + Gymnasium
pip install learnlens-rl[sb3]         # + Stable Baselines3
pip install learnlens-rl[rllib]       # + Ray RLlib
pip install learnlens-rl[all]         # everything
"""

from learnlens.config import LensConfig
from learnlens.scorer import LQSReport, compute_lqs, make_report
from learnlens.wrapper import LensWrapper

__all__ = [
    "LensWrapper",
    "LensConfig",
    "LQSReport",
    "compute_lqs",
    "make_report",
]

__version__ = "0.2.0"
__author__ = "Ajay Bandiwaddar"