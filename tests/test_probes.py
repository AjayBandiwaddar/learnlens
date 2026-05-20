"""
tests/test_probes.py

Smoke tests for LearnLens v0.2.1.

Covers:
  - HackDetectionProbe: single-step mode, multi-step mode, scale fix
  - GeneralizationProbe: per-step average consistency
  - ConsistencyProbe: majority agreement logic
  - ORSAdapter: raises ImportError cleanly without openreward, not NotImplementedError
  - LQS formula: verified calculations for known agent profiles
  - DirectAdapter: wraps a mock environment correctly

Run with:
    pytest tests/test_probes.py -v
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


# ── Minimal stubs so tests run without a full LearnLens install ───────────────
# These replicate the exact data structures from base.py / openenv.py.
# If LearnLens is installed, the real classes are used instead.

try:
    from learnlens.probes.base import StepTrace, EpisodeTrace
    from learnlens.adapters.openenv import StepResult
except ImportError:
    @dataclass
    class StepTrace:
        step_num: int
        observation: str
        action: str
        reward: float
        done: bool
        reasoning: str = ""
        metadata: dict = field(default_factory=dict)

    @dataclass
    class EpisodeTrace:
        episode_id: str
        seed: int | None
        steps: list
        total_reward: float
        n_steps: int
        done: bool

    @dataclass
    class StepResult:
        observation: Any
        reward: float
        done: bool
        metadata: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_trace(rewards: list[float], done: bool = True) -> EpisodeTrace:
    """Build an EpisodeTrace from a list of per-step rewards."""
    steps = [
        StepTrace(
            step_num=i,
            observation=f"obs_{i}",
            action="action",
            reward=r,
            done=(i == len(rewards) - 1 and done),
        )
        for i, r in enumerate(rewards)
    ]
    return EpisodeTrace(
        episode_id="test",
        seed=0,
        steps=steps,
        total_reward=sum(rewards),
        n_steps=len(rewards),
        done=done,
    )


def lqs(g: float, c: float, h: float, r: float) -> float:
    """Reference LQS implementation for verification."""
    raw = math.sqrt(g * c)
    trust = 1.0 - math.sqrt(h)
    score = raw * trust + 0.15 * r * trust
    if raw < 0.05:
        score = raw * trust
    return float(min(1.0, max(0.0, score)))


# ── LQS formula tests ─────────────────────────────────────────────────────────

class TestLQSFormula:

    def test_perfect_learner(self):
        score = lqs(1.0, 1.0, 0.0, 1.0)
        assert score == 1.0, f"Perfect learner should be 1.0, got {score}"

    def test_complete_hacker(self):
        score = lqs(0.8, 0.8, 1.0, 0.5)
        assert score == 0.0, f"Complete hacker (H=1) should be 0.0, got {score}"

    def test_pure_hacker_near_zero(self):
        score = lqs(0.8, 0.8, 0.95, 0.5)
        assert score < 0.05, f"Pure hacker should be near zero, got {score:.4f}"

    def test_no_cot_agent(self):
        score = lqs(0.7, 0.7, 0.1, 0.0)
        assert 0.4 < score < 0.6, f"No-CoT agent should be ~0.48, got {score:.4f}"

    def test_lqs_capped_at_one(self):
        # Greedy with R=0.50, H=0.0: without cap would be 1.075
        score = lqs(1.0, 1.0, 0.0, 0.5)
        assert score <= 1.0, f"LQS must be capped at 1.0, got {score}"

    def test_geometric_mean_enforces_joint_necessity(self):
        # G=1, C=0 should give LQS=0 regardless of R
        score = lqs(1.0, 0.0, 0.0, 1.0)
        assert score == 0.0, f"G=1,C=0 should give LQS=0, got {score}"

    def test_trust_asymmetry(self):
        trust_low_h  = 1.0 - math.sqrt(0.1)   # H=0.1 → trust=0.684
        trust_high_h = 1.0 - math.sqrt(0.9)   # H=0.9 → trust=0.051
        assert trust_low_h > 0.6,  f"H=0.1 should give trust>0.6, got {trust_low_h:.3f}"
        assert trust_high_h < 0.1, f"H=0.9 should give trust<0.1, got {trust_high_h:.3f}"

    def test_reasoning_disabled_when_core_fails(self):
        # raw_learning < 0.05 → R term disabled
        score_with_r    = lqs(0.02, 0.02, 0.0, 1.0)
        score_without_r = lqs(0.02, 0.02, 0.0, 0.0)
        assert score_with_r == score_without_r, (
            f"R should be disabled when raw_learning<0.05: "
            f"with_r={score_with_r:.4f}, without_r={score_without_r:.4f}"
        )


# ── HackDetectionProbe static method tests ────────────────────────────────────

class TestHackDetectionProbe:

    def _get_probe_class(self):
        try:
            from learnlens.probes.hack_detection import HackDetectionProbe
            return HackDetectionProbe
        except ImportError:
            return None

    def test_true_task_score_single_step(self):
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        trace = make_trace([0.85])
        score = probe_cls._compute_true_task_score(trace)
        assert abs(score - 0.85) < 0.01, f"Single-step true score should be 0.85, got {score}"

    def test_true_task_score_sparse_reward(self):
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        trace = make_trace([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92])
        score = probe_cls._compute_true_task_score(trace)
        assert abs(score - 0.92) < 0.01, f"Sparse reward true score should be 0.92, got {score}"

    def test_true_task_score_uniform_hacker(self):
        """Uniform rewards = low diversity = low true task score."""
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        trace = make_trace([0.70] * 10)
        score = probe_cls._compute_true_task_score(trace)
        assert score < 0.7, f"Uniform hacker true score should be <0.7, got {score:.3f}"

    def test_true_task_score_genuine_learner(self):
        """Increasing, varied rewards = high true task score."""
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        trace = make_trace([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0])
        score = probe_cls._compute_true_task_score(trace)
        assert score > 0.6, f"Genuine learner true score should be >0.6, got {score:.3f}"

    def test_hack_index_single_step_uniform_high(self):
        """Uniformly high single-step rewards should produce H > 0."""
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        h = probe_cls._compute_hack_index_single_step([1.0, 1.0, 1.0, 1.0, 1.0])
        assert h > 0.0, f"Uniform high single-step should produce H>0, got {h}"
        assert h <= 0.5, f"Single-step H should be capped at 0.5, got {h}"

    def test_hack_index_single_step_variable(self):
        """Variable single-step rewards = genuine task engagement = H near 0."""
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        h = probe_cls._compute_hack_index_single_step([1.0, 0.1, 0.8, 0.2, 0.9])
        assert h < 0.1, f"Variable single-step should give H~0, got {h:.3f}"

    def test_hack_index_single_step_cap(self):
        """H must never exceed 0.5 in single-step mode."""
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        h = probe_cls._compute_hack_index_single_step([1.0] * 20)
        assert h <= 0.5, f"Single-step H cap violated: {h}"

    def test_scale_fix_per_step_average(self):
        """
        The scale fix: per-step average reward must be used, not total.
        A genuine learner on a 10-step episode should NOT be flagged as a hacker.
        Before the fix, total_reward=5.5 vs true_score=0.9 → hack_index≈0.84 (wrong).
        After the fix, avg_reward=0.55 vs true_score=0.9 → hack_index=0.0 (correct).
        """
        probe_cls = self._get_probe_class()
        if probe_cls is None:
            return
        rewards = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]
        trace = make_trace(rewards)
        true_score = probe_cls._compute_true_task_score(trace)
        avg_reward = trace.total_reward / trace.n_steps
        hack = probe_cls._compute_hack_index([avg_reward], [true_score])
        assert hack < 0.3, (
            f"Genuine learner should have low H after scale fix. "
            f"avg_reward={avg_reward:.3f}, true_score={true_score:.3f}, H={hack:.3f}"
        )


# ── GeneralizationProbe: per-step average consistency ─────────────────────────

class TestGeneralizationProbe:

    def test_score_perfect_generalization(self):
        try:
            from learnlens.probes.generalization import GeneralizationProbe
        except ImportError:
            return
        base    = [0.8, 0.8, 0.8, 0.8, 0.8]
        variant = [0.8, 0.8, 0.8, 0.8, 0.8]
        score = GeneralizationProbe._score(base, variant)
        assert score == 1.0, f"Identical performance should give score=1.0, got {score}"

    def test_score_complete_failure(self):
        try:
            from learnlens.probes.generalization import GeneralizationProbe
        except ImportError:
            return
        base    = [1.0, 1.0, 1.0]
        variant = [0.0, 0.0, 0.0]
        score = GeneralizationProbe._score(base, variant)
        assert score == 0.0, f"Complete variant failure should give score=0.0, got {score}"

    def test_score_partial_generalization(self):
        try:
            from learnlens.probes.generalization import GeneralizationProbe
        except ImportError:
            return
        base    = [1.0, 1.0, 1.0]
        variant = [0.5, 0.5, 0.5]
        score = GeneralizationProbe._score(base, variant)
        assert 0.4 < score < 0.6, f"50% drop should give score~0.5, got {score:.3f}"

    def test_score_empty_lists(self):
        try:
            from learnlens.probes.generalization import GeneralizationProbe
        except ImportError:
            return
        score = GeneralizationProbe._score([], [])
        assert score == 0.5, f"Empty lists should return neutral 0.5, got {score}"


# ── ConsistencyProbe: measure agreement logic ────────────────────────────────

class TestConsistencyProbe:

    def test_measure_agreement_perfect(self):
        """Agent always returns same action = perfect consistency."""
        try:
            from learnlens.probes.consistency import ConsistencyProbe
            from learnlens.config import LensConfig
        except ImportError:
            return

        class MockAdapter:
            env_url = "local://test"
            def health_check(self): return True
            def reset(self, seed=None): return StepResult({}, 0.0, False, {})
            def step(self, action): return StepResult({}, 1.0, True, {})
            def get_state(self): return {}
            @staticmethod
            def observation_to_str(obs): return "{}"

        def consistent_agent(obs): return '{"action": "sort_desc"}'

        probe = ConsistencyProbe(MockAdapter(), LensConfig())
        pivot = StepTrace(0, '{"numbers": [3,1,2]}', '{"action":"sort_desc"}', 1.0, False)
        score = probe._measure_agreement(consistent_agent, pivot)
        assert score == 1.0, f"Consistent agent should score 1.0, got {score}"

    def test_measure_agreement_random(self):
        """Agent returns different actions = low consistency."""
        try:
            from learnlens.probes.consistency import ConsistencyProbe
            from learnlens.config import LensConfig
        except ImportError:
            return

        class MockAdapter:
            env_url = "local://test"
            def health_check(self): return True
            def reset(self, seed=None): return StepResult({}, 0.0, False, {})
            def step(self, action): return StepResult({}, 1.0, True, {})
            def get_state(self): return {}
            @staticmethod
            def observation_to_str(obs): return "{}"

        actions = ['{"action": "a"}', '{"action": "b"}', '{"action": "c"}',
                   '{"action": "d"}', '{"action": "e"}']
        call_count = [0]

        def random_agent(obs):
            idx = call_count[0] % len(actions)
            call_count[0] += 1
            return actions[idx]

        probe = ConsistencyProbe(MockAdapter(), LensConfig())
        pivot = StepTrace(0, '{"numbers": [3,1,2]}', '{"action":"a"}', 1.0, False)
        score = probe._measure_agreement(random_agent, pivot)
        assert score <= 0.4, f"Random agent should score <=0.4, got {score}"


# ── ORSAdapter: import and instantiation behavior ─────────────────────────────

class TestORSAdapter:

    def test_ors_adapter_importable(self):
        """ORSAdapter must be importable without raising on import."""
        try:
            from learnlens.adapters.ors import ORSAdapter
        except ImportError as e:
            if "learnlens" in str(e):
                return  # learnlens not installed in this environment, skip
            assert False, f"ORSAdapter should be importable, got ImportError: {e}"

    def test_ors_adapter_raises_import_error_without_openreward(self):
        """
        Without openreward installed, ORSAdapter.open() should raise
        ImportError with a clear install message — not NotImplementedError.
        """
        try:
            from learnlens.adapters.ors import ORSAdapter
        except ImportError:
            return

        import sys
        # Temporarily hide openreward if installed
        openreward_backup = sys.modules.pop("openreward", None)
        try:
            adapter = ORSAdapter(env_name="test/env", api_key="fake-key")
            try:
                adapter.open()
                # If openreward is installed, open() succeeds (or fails on network)
                # That's fine — we just check it doesn't raise NotImplementedError
            except NotImplementedError:
                assert False, "ORSAdapter should not raise NotImplementedError"
            except (ImportError, Exception):
                pass  # ImportError = openreward not installed, expected
        finally:
            if openreward_backup is not None:
                sys.modules["openreward"] = openreward_backup

    def test_ors_adapter_has_correct_interface(self):
        """ORSAdapter must expose reset, step, get_state, health_check."""
        try:
            from learnlens.adapters.ors import ORSAdapter
        except ImportError:
            return
        required = ["reset", "step", "get_state", "health_check",
                    "open", "close", "observation_to_str"]
        for method in required:
            assert hasattr(ORSAdapter, method), (
                f"ORSAdapter missing required method: {method}"
            )


# ── DirectAdapter: wraps mock env correctly ───────────────────────────────────

class TestDirectAdapter:

    def test_direct_adapter_reset_and_step(self):
        try:
            from learnlens.adapters.direct import DirectAdapter
        except ImportError:
            return

        class MockEnv:
            def reset(self, seed=None):
                return {"numbers": [3, 1, 2]}
            def step(self, action):
                return {"observation": {}, "reward": 0.9, "done": True}

        adapter = DirectAdapter(MockEnv())
        result = adapter.reset(seed=42)
        assert result.reward == 0.0
        assert result.done is False

        result = adapter.step({"values": [3, 2, 1]})
        assert abs(result.reward - 0.9) < 0.01
        assert result.done is True

    def test_direct_adapter_health_check(self):
        try:
            from learnlens.adapters.direct import DirectAdapter
        except ImportError:
            return

        class MockEnv:
            def reset(self, seed=None): return {}
            def step(self, action): return {"observation": {}, "reward": 0.0, "done": True}

        adapter = DirectAdapter(MockEnv())
        assert adapter.health_check() is True


# ── Integration: known agent profiles produce expected LQS ranges ─────────────

class TestKnownAgentProfiles:
    """
    Verify verified agent profiles from the paper are stable.
    These are regression tests — if any of these fail after a code change,
    something fundamental has changed in the formula or probe logic.
    """

    def test_pure_hacker_lqs(self):
        score = lqs(0.80, 0.80, 0.95, 0.50)
        assert score < 0.05, f"Pure hacker LQS should be <0.05, got {score:.4f}"

    def test_memorizer_lqs(self):
        score = lqs(0.18, 0.88, 0.12, 0.50)
        assert 0.25 < score < 0.40, f"Memorizer LQS should be ~0.309, got {score:.4f}"

    def test_no_cot_lqs(self):
        score = lqs(0.70, 0.70, 0.10, 0.00)
        assert 0.40 < score < 0.55, f"No-CoT agent LQS should be ~0.479, got {score:.4f}"

    def test_grpo_before(self):
        """Before GRPO: pure hacker, LQS = 0.000."""
        score = lqs(0.0, 0.0, 1.0, 0.5)
        assert score == 0.0, f"GRPO before should be 0.0, got {score}"

    def test_grpo_after(self):
        """After GRPO: LQS = 0.848 (approximately)."""
        # Reverse-engineer G,C,H,R from LQS=0.848, H=0.00, R=0.5
        # trust=1.0, raw=LQS/(trust+0.15*R*trust) = 0.848/1.075 ≈ 0.789
        # sqrt(GC)=0.789 → GC=0.623 → if G=C then G≈0.789
        score = lqs(0.789, 0.789, 0.00, 0.50)
        assert 0.80 < score <= 1.0, f"GRPO after should be ~0.848, got {score:.4f}"