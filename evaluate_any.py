"""
evaluate_any.py  --  Run LearnLens against ANY OpenEnv environment by URL.

Usage:
    python evaluate_any.py https://team-space.hf.space
    python evaluate_any.py https://team-space.hf.space --episodes 3 --no-reasoning
    python evaluate_any.py https://team-space.hf.space --groq-key gsk_...
"""

import argparse
import json
import sys
import time

from learnlens import LensWrapper, LensConfig
from learnlens.adapters.openenv import OpenEnvAdapter


# ── Observation inspector ──────────────────────────────────────────────────

def inspect_env(url: str) -> dict:
    """
    Open ONE WebSocket session, peek at observation, close it.
    Returns observation dict for agent to use.
    """
    try:
        adapter = OpenEnvAdapter(env_url=url)
        adapter.open()
        result  = adapter.reset(seed=0)
        obs     = result.observation
        adapter.close()
        return obs if isinstance(obs, dict) else {}
    except Exception:
        return {}


# ── Generic agent ──────────────────────────────────────────────────────────

# Global so agent can access original numbers from observation
_last_original: list = []
_last_task: str = "easy"


def generic_agent(obs_str: str) -> tuple[str, str]:
    """
    Best-effort agent for unknown environments.
    Handles: NumberSort (with original field), Queue Doctor, generic JSON.
    Returns (action_str, reasoning_str).
    """
    global _last_original, _last_task

    try:
        obs = json.loads(obs_str)
    except (json.JSONDecodeError, ValueError):
        return json.dumps({"action": obs_str[:200]}), "Plain text obs"

    # NumberSort-style: {"numbers": [...], "task": "...", "original": [...]}
    if "numbers" in obs and isinstance(obs["numbers"], list):
        numbers  = obs["numbers"]
        task     = obs.get("task", "easy")
        original = obs.get("original", numbers)

        _last_original = original
        _last_task     = task

        if task == "hard":
            evens  = sorted([x for x in numbers if x % 2 == 0], reverse=True)
            odds   = sorted([x for x in numbers if x % 2 != 0], reverse=True)
            answer = evens + odds
        else:
            answer = sorted(numbers, reverse=True)

        reasoning = f"NumberSort env. Task={task}. Sorting {len(numbers)} numbers."

        # Include original field if the env expects it
        action = {"values": answer}
        if original:
            action["original"] = original
        if task:
            action["task"] = task

        return json.dumps(action), reasoning

    # Queue Doctor-style: {"queue": [...]}
    if "queue" in obs and isinstance(obs["queue"], list):
        queue = obs["queue"]
        if queue:
            best = min(queue, key=lambda p: p.get("severity", 99))
            pid  = best.get("patient_id", best.get("id", "P001"))
            reasoning = f"Queue env. Serving {pid} (sev={best.get('severity')})."
            return json.dumps({"patient_id": pid}), reasoning

    # Generic: act on first list field
    for key, val in obs.items():
        if isinstance(val, list) and val:
            return json.dumps({"action": str(val[0])}), f"Acting on field '{key}'"

    return json.dumps({"action": "next"}), "Unknown structure"


# ── Health check via HTTP only (no WebSocket) ──────────────────────────────

def check_health_http(url: str, retries: int = 3, delay: float = 5.0) -> bool:
    """
    Health check via HTTP GET /health only.
    Does NOT open a WebSocket session — preserves capacity for evaluation.
    """
    try:
        import httpx
    except ImportError:
        print("  httpx not installed. pip install httpx")
        return False

    for attempt in range(retries):
        try:
            resp = httpx.get(f"{url.rstrip('/')}/health", timeout=15)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        if attempt < retries - 1:
            print(f"  Retrying ({attempt+2}/{retries}) in {delay}s...")
            time.sleep(delay)
    return False


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LearnLens: LQS evaluation against any OpenEnv Space."
    )
    parser.add_argument("env_url", help="Full HF Space URL")
    parser.add_argument("--episodes", "-e", type=int, default=3)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--groq-key", default=None)
    args = parser.parse_args()

    url = args.env_url.rstrip("/")

    print()
    print("=" * 60)
    print("  LearnLens -- Universal Environment Evaluation")
    print("=" * 60)
    print(f"  Target   : {url}")
    print(f"  Episodes : {args.episodes} per probe")
    print()

    # ── Step 1: HTTP health check (no WebSocket used) ──────────────────
    print("  Checking environment health (HTTP)...")
    if not check_health_http(url):
        print(f"\n  ERROR: {url} is not responding.")
        sys.exit(1)
    print("  Environment is live.\n")

    # ── Step 2: Peek at observation format (one quick WS session) ──────
    print("  Inspecting observation format...")
    sample_obs = inspect_env(url)
    if sample_obs:
        preview = json.dumps(sample_obs, indent=2)[:400].replace("\n", " ")
        print(f"  Preview: {preview}\n")
    else:
        print("  Could not inspect observation. Proceeding anyway.\n")

    # Small delay to ensure the inspection session fully closes
    time.sleep(2)

    # ── Step 3: Full evaluation (single persistent WS session) ─────────
    config = LensConfig(
        run_reasoning=not args.no_reasoning,
        max_steps_per_episode=10,   # short for unknown envs
    )

    judge_model   = "llama-3.1-8b-instant" if args.groq_key else "claude-sonnet-4-20250514"
    judge_api_key = args.groq_key

    wrapper = LensWrapper(
        env_url=url,
        config=config,
        judge_model=judge_model,
        judge_api_key=judge_api_key,
    )

    print("  Running probes: Generalization, Consistency, HackDetection",
          "(+ Reasoning)" if not args.no_reasoning else "")
    print()

    try:
        report = wrapper.evaluate(
            agent_fn=generic_agent,
            n_episodes=args.episodes,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print()
        print("  If you see CAPACITY_REACHED: wait 10s and retry.")
        print("  The HF Space allows 1 concurrent session.")
        print("  Run: python evaluate_any.py <url> --no-reasoning --episodes 2")
        sys.exit(1)

    report.print_report()

    print()
    print("  pip install learnlens-rl")
    print("  github.com/AjayBandiwaddar/learnlens")
    print()


if __name__ == "__main__":
    main()