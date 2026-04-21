"""
learnlens/adapters/ors.py

ORSAdapter — Phase 2 stub.
ORS (Open Reward Standard) adapter for 330+ environments at openrewardstandard.io.

TODO Phase 2:
- reset() via ORS /start endpoint
- step() via ORS tool-calling HTTP protocol
- Map ORS observations to LearnLens EpisodeTrace format
- Test against EnvCommons environments
"""


class ORSAdapter:
    """Phase 2 stub. Raises NotImplementedError on instantiation."""

    def __init__(self, ors_endpoint: str):
        raise NotImplementedError(
            "ORSAdapter is Phase 2. OpenEnv environments are fully supported. "
            "ORS support coming post-hackathon."
        )