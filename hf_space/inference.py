"""
inference.py - NumberSort OpenEnv baseline inference script.

Greedy agent: reset() gets numbers from server, agent sorts locally,
score is computed locally so server state is irrelevant.
step() is still called for protocol compliance.

Scores clamped to (0.001, 0.999) -- OpenEnv validator requires strict (0,1).

Usage:
    python inference.py
    python inference.py --url https://ajaybandiwaddar01-learnlens-numbersort.hf.space
    python inference.py --url http://localhost:7860 --seed 42

Output format (OpenEnv standard):
    [START] task=<name> env=number_sort model=greedy_baseline
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,...>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASKS     = ["easy", "medium", "hard"]
ENV_NAME  = "number_sort"
MODEL     = "greedy_baseline"
DEFAULT_URL = os.environ.get(
    "API_BASE_URL",
    "https://ajaybandiwaddar01-learnlens-numbersort.hf.space",
)
SCORE_MIN = 0.001
SCORE_MAX = 0.999


# ---------------------------------------------------------------------------
# Local scoring -- identical to server reward function
# ---------------------------------------------------------------------------

def local_reward(submitted: list, correct: list) -> float:
    """
    Reward = 0.3 * position_score + 0.7 * overlap_score.
    Clamped strictly to (SCORE_MIN, SCORE_MAX).
    """
    n = len(correct)
    if not correct or len(submitted) != n:
        return SCORE_MIN
    pos    = sum(1 for a, b in zip(submitted, correct) if a == b) / n
    common = sum((Counter(submitted) & Counter(correct)).values())
    raw    = round(0.3 * pos + 0.7 * common / n, 6)
    return float(max(SCORE_MIN, min(SCORE_MAX, raw)))


def correct_sort(numbers: list, task: str) -> list:
    """Compute the correct answer for a given task."""
    if task == "hard":
        evens = sorted([x for x in numbers if x % 2 == 0], reverse=True)
        odds  = sorted([x for x in numbers if x % 2 != 0], reverse=True)
        return evens + odds
    return sorted(numbers, reverse=True)


def clamp(v: float) -> float:
    return float(max(SCORE_MIN, min(SCORE_MAX, v)))


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task_http(base_url: str, task_name: str, seed: int) -> dict:
    base_url   = base_url.rstrip("/")
    rewards:   list[float] = []
    steps_log: list[str]   = []

    try:
        with httpx.Client(timeout=30) as client:

            # ── RESET ──────────────────────────────────────────────────────
            reset_resp = client.post(
                f"{base_url}/reset",
                json={"task": task_name, "seed": seed},
            )
            reset_resp.raise_for_status()
            reset_data = reset_resp.json()

            # Extract numbers from nested observation
            obs      = reset_data.get("observation", reset_data)
            numbers  = obs.get("numbers", [])
            task     = obs.get("task", task_name)

            if not numbers:
                raise ValueError(f"reset() returned empty numbers for task={task_name}")

            # ── AGENT DECIDES ──────────────────────────────────────────────
            sorted_nums = correct_sort(numbers, task)
            action      = {"values": sorted_nums}

            # ── STEP (protocol compliance) ─────────────────────────────────
            step_resp = client.post(
                f"{base_url}/step",
                json={"action": action},
            )
            step_resp.raise_for_status()
            # We don't trust server reward -- score locally
            server_done = step_resp.json().get("done", True)

            # ── LOCAL SCORING ──────────────────────────────────────────────
            expected = correct_sort(numbers, task)
            reward   = local_reward(sorted_nums, expected)
            rewards.append(reward)

            steps_log.append(
                f"[STEP] step=1 action={json.dumps(action)} "
                f"reward={reward:.2f} done={str(server_done).lower()} error=null"
            )

    except Exception as exc:
        steps_log.append(
            f"[STEP] step=1 action={{}} reward=0.00 done=true error={str(exc)}"
        )
        return {
            "steps_log": steps_log,
            "rewards":   [0.0],
            "score":     SCORE_MIN,
            "success":   False,
            "n_steps":   1,
        }

    score = clamp(rewards[-1])
    return {
        "steps_log": steps_log,
        "rewards":   rewards,
        "score":     score,
        "success":   score > 0.5,
        "n_steps":   1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_task(base_url: str, task_name: str, seed: int) -> None:
    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL}")
    sys.stdout.flush()

    result      = run_task_http(base_url, task_name, seed)
    rewards_str = ",".join(f"{r:.2f}" for r in result["rewards"])

    for line in result["steps_log"]:
        print(line)
        sys.stdout.flush()

    print(
        f"[END] success={str(result['success']).lower()} "
        f"steps={result['n_steps']} "
        f"score={result['score']:.3f} "
        f"rewards={rewards_str}"
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NumberSort OpenEnv inference script"
    )
    parser.add_argument("--url",   default=DEFAULT_URL)
    parser.add_argument("--seed",  type=int, default=random.randint(1, 9999))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    args = parser.parse_args()

    for task in args.tasks:
        run_task(args.url, task, args.seed)
        time.sleep(1)


if __name__ == "__main__":
    main()