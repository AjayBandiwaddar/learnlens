"""
demo.py  LearnLens end-to-end demonstration.

Run:
    python demo.py               # easy task, 5 episodes
    python demo.py medium 8      # medium task, 8 episodes

Three agents against NumberSort (reward = 0.3*pos + 0.7*overlap):

  greedy_agent   sorts correctly descending          reward=1.00  LQS=1.00
  hacking_agent  ascending order + brittle parsing   reward=0.70  LQS<greedy
  random_agent   random order                        reward~0.75  LQS=0.50

Key story:
  Random reward (0.75) > Hacking reward (0.70) -- reward gets it backwards.
  LQS correctly ranks: greedy > hacking > random.
  Reward is not learning. LearnLens measures learning.

No API key needed (reasoning probe disabled).
No network needed (DirectAdapter wraps local environment).
"""

import json
import random

from learnlens import LensWrapper, LensConfig
from learnlens.adapters.direct import DirectAdapter
from learnlens.envs.number_sort.environment import NumberSortEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Agent definitions
# ─────────────────────────────────────────────────────────────────────────────

def greedy_agent(obs_str: str) -> str:
    """
    Genuine learner: robustly parses the observation and sorts correctly.
    Works on all 5 ConsistencyProbe paraphrase templates (extracts JSON
    from anywhere in the string).
    """
    obs = _parse_obs_robust(obs_str)
    numbers = obs.get("numbers", [])
    task    = obs.get("task", "easy")

    if task == "hard":
        evens  = sorted([x for x in numbers if x % 2 == 0], reverse=True)
        odds   = sorted([x for x in numbers if x % 2 != 0], reverse=True)
        answer = evens + odds
    else:
        answer = sorted(numbers, reverse=True)

    return json.dumps({"values": answer})


def hacking_agent(obs_str: str) -> str:
    """
    Reward hacker with two intentional flaws:

    Flaw 1 — WRONG DIRECTION:
        Sorts ascending instead of descending.
        Exploits the overlap term (weight 0.7): any permutation of the
        correct numbers gets overlap=1.0, so reward = 0.3*0 + 0.7*1.0 = 0.70.
        High reward. Wrong answer.

    Flaw 2 — BRITTLE PARSING:
        Only handles raw JSON. If the observation is wrapped in any
        paraphrase template ("Current state: {...}", "Step 3:\n{...}", etc.)
        the strict json.loads fails and returns an empty list.
        ConsistencyProbe surfaces this: 4 out of 5 phrasings break the agent.

    This models real reward hackers: exploit one pattern in the reward signal
    and break under any distribution shift.
    """
    obs_str = obs_str.strip()

    # BRITTLE: requires observation to START with "{"
    if not obs_str.startswith("{"):
        return json.dumps({"values": []})   # Paraphrased template -- agent breaks

    try:
        obs     = json.loads(obs_str)       # Strict parse, no fallback
        numbers = obs["numbers"]            # Hard key access
        return json.dumps({"values": sorted(numbers)})   # ASCENDING -- wrong!
    except (json.JSONDecodeError, KeyError, TypeError):
        return json.dumps({"values": []})


def random_agent(obs_str: str) -> str:
    """
    Baseline: submits numbers in random order.
    Still scores ~0.70-0.75 reward because overlap=1.0 for any permutation.
    ConsistencyProbe: gives different answers each time -> consistency~0.20.
    """
    obs     = _parse_obs_robust(obs_str)
    numbers = list(obs.get("numbers", []))
    random.shuffle(numbers)
    return json.dumps({"values": numbers})


# ─────────────────────────────────────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(task: str = "easy", n_episodes: int = 5) -> None:
    _header(task, n_episodes)

    env     = NumberSortEnvironment(task=task)
    adapter = DirectAdapter(env, env_url=f"local://number_sort/{task}")
    config  = LensConfig(run_reasoning=False)   # no API key needed
    wrapper = LensWrapper(adapter=adapter, config=config)

    agents = [
        ("Greedy Agent   (correct sort, robust parsing)",  greedy_agent),
        ("Hacking Agent  (ascending sort, brittle parse)", hacking_agent),
        ("Random Agent   (baseline)",                      random_agent),
    ]

    reports = {}
    for label, agent_fn in agents:
        print(f"\n-- Evaluating: {label} --")
        report = wrapper.evaluate(agent_fn=agent_fn, n_episodes=n_episodes)
        reports[label] = report
        report.print_report()

    _summary(reports)
    _probe_explanation()
    _pitch()


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(task: str, n_episodes: int) -> None:
    print()
    print("=" * 62)
    print("  LearnLens -- End-to-End Demo")
    print(f"  Task        : NumberSort ({task})")
    print(f"  Episodes    : {n_episodes} per probe")
    print()
    print("  Reward fn   : 0.3 x pos_score + 0.7 x overlap")
    print("  Loophole    : overlap=1.0 for any permutation -> reward >= 0.7")
    print("=" * 62)


def _summary(reports: dict) -> None:
    print()
    print("=" * 62)
    print("  Summary")
    print("=" * 62)
    print(f"  {'Agent':<44}  {'Reward':>6}  {'LQS':>6}")
    print(f"  {'-'*44}  {'------':>6}  {'------':>6}")
    for label, r in reports.items():
        name = label.split("(")[0].strip()
        flag = "  <-- FLAGGED" if r.hack_flagged else ""
        print(f"  {name:<44}  {r.mean_reward:>6.2f}  {r.lqs:>6.2f}{flag}")
    print()
    print("  Key insight:")
    print("  Random reward (~0.75) > Hacking reward (0.70) -- reward is wrong.")
    print("  LQS correctly ranks: Greedy > Hacking > Random.")
    print()


def _probe_explanation() -> None:
    print("=" * 62)
    print("  What each probe reveals")
    print("=" * 62)
    print()
    print("  GeneralizationProbe:")
    print("    Seeds 0-4 vs seeds 1000-1004.")
    print("    All agents perform the same across seeds.")
    print("    Probe score ~ 1.0 for all -- can't distinguish on this alone.")
    print()
    print("  ConsistencyProbe:  <-- catches the hacker")
    print("    Same observation, 5 different phrasings.")
    print("    Greedy  : robust JSON extraction -> correct answer always -> 1.0")
    print("    Hacking : strict JSON-only -> fails on 4/5 phrasings -> < 1.0")
    print("    Random  : shuffles differently each call -> 0.20")
    print("    -> Reveals brittleness and inconsistency.")
    print()
    print("  HackDetectionProbe:")
    print("    NumberSort is single-step, so signal is limited here.")
    print("    Queue Doctor (true multi-step MDP) gives strong hack signal.")
    print("    Probe shows full power on real RL environments.")
    print()
    print("  LQS = sqrt(G x C) x (1 - sqrt(H)) + 0.15 x R x trust")
    print("    Low consistency collapses LQS even with high reward.")
    print()


def _pitch() -> None:
    print("=" * 62)
    print("  The Pitch")
    print("=" * 62)
    print()
    print("  Every team here measured REWARD.")
    print("  I measured LEARNING.")
    print()
    print("  The random agent scored 0.75 reward.")
    print("  The hacking agent scored 0.70 reward.")
    print("  Reward says: random > hacker.")
    print("  LQS says:    hacker > random (hacker is at least consistent).")
    print("  Both are wrong, but reward can't even rank them correctly.")
    print()
    print("  On Queue Doctor (true multi-step RL env from Round 1):")
    print("    An agent gaming the triage grader: reward=0.73.")
    print("    LearnLens: LQS=0.27. Verdict: reward hacking.")
    print()
    print("  pip install learnlens")
    print("  Three lines. Any OpenEnv environment.")
    print("=" * 62)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Observation parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_obs_robust(obs_str: str) -> dict:
    """
    Robust parser: handles raw JSON and all 5 ConsistencyProbe templates.
    Extracts the embedded JSON dict from any wrapping text.
    """
    if isinstance(obs_str, dict):
        return obs_str

    s = obs_str.strip()

    if s.startswith("{"):
        try:
            return json.loads(s)
        except (ValueError, TypeError):
            pass

    # Extract JSON object embedded in paraphrase text
    start = s.find("{")
    end   = s.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(s[start:end])
        except (ValueError, TypeError):
            pass

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "easy"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_demo(task=task, n_episodes=n_ep)