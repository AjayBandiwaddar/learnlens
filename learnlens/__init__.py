"""
learnlens -- Universal evaluation layer for OpenEnv agentic RL environments.

Measures WHAT an agent learned, not just HOW MUCH reward it accumulated.

Quick start:
    from learnlens import LensWrapper

    env = LensWrapper(env_url="https://your-openenv-space.hf.space")
    results = env.evaluate(agent_fn=my_agent)
    results.print_report()
    print(results.lqs)   # Learning Quality Score in [0.0, 1.0]

pip install learnlens
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

__version__ = "0.1.0"
__author__ = "Ajay Bandiwaddar"