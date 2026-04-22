"""
learnlens/probes/reasoning.py

ReasoningProbe: measures whether the agent chain-of-thought aligns with
its actions and references the environment state accurately.

Judge model MUST differ from agent model (MT-Bench methodology).
Default judge: claude-sonnet-4-6 via Anthropic API.
Only probe requiring ANTHROPIC_API_KEY.
Returns 0.5 (neutral) if no CoT captured or no API key.

Scoring dimensions (0.0-1.0 each):
  Relevance:   agent referenced key state variables
  Coherence:   reasoning logically supports the action
  Uncertainty: agent flagged ambiguity when appropriate
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from learnlens.probes.base import BaseProbe, StepTrace

_JUDGE_PROMPT = """\
You are evaluating the reasoning quality of an AI agent interacting with \
a reinforcement learning environment.

Agent observation:
{observation}

Agent reasoning (chain-of-thought):
{reasoning}

Agent action taken:
{action}

Reward received after this action:
{reward}

Score the reasoning quality on THREE dimensions, each from 0.0 to 1.0:

1. Relevance (0-1): Did the agent explicitly reference key state variables?
   0 = ignored observation, 1 = referenced all relevant features.

2. Coherence (0-1): Does reasoning logically justify the chosen action?
   0 = reasoning contradicts action, 1 = reasoning clearly supports action.

3. Uncertainty (0-1): Did agent appropriately acknowledge ambiguity?
   0 = overconfident on ambiguous states, 1 = appropriate calibration.

Respond ONLY with JSON, no preamble:
{{"relevance": <float>, "coherence": <float>, "uncertainty": <float>}}\
"""

STEPS_PER_EPISODE = 3
MAX_RETRIES = 2
RETRY_DELAY_S = 2


class ReasoningProbe(BaseProbe):
    """
    LLM judge evaluates agent chain-of-thought quality.
    Judge is always a different model from the agent.
    Returns 0.5 if no CoT captured or no API key available.
    """

    def __init__(self, adapter, config, judge_model: str, judge_api_key: str | None = None) -> None:
        super().__init__(adapter, config)
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")

    def evaluate(
        self,
        agent_fn: Callable[[str], str | tuple[str, str]],
        n_episodes: int = 5,
    ) -> float:
        if not self.judge_api_key:
            return 0.5

        all_scores: list[float] = []

        for i in range(n_episodes):
            trace = self._run_episode(agent_fn, seed=i)
            for step in trace.steps[:STEPS_PER_EPISODE]:
                if not step.reasoning:
                    continue
                score = self._judge_step(step)
                if score is not None:
                    all_scores.append(score)

        return float(self._safe_mean(all_scores)) if all_scores else 0.5

    def _judge_step(self, step: StepTrace) -> float | None:
        prompt = _JUDGE_PROMPT.format(
            observation=step.observation[:2000],
            reasoning=step.reasoning[:1000],
            action=step.action[:200],
            reward=f"{step.reward:.4f}",
        )
        raw = self._call_judge_api(prompt)
        return self._parse_judge_response(raw) if raw else None

    def _call_judge_api(self, prompt: str) -> str | None:
        # Auto-detect provider from model name
        if "gpt" in self.judge_model or "o1" in self.judge_model or "o3" in self.judge_model or "llama" in self.judge_model or "mixtral" in self.judge_model or "gemma" in self.judge_model:
            return self._call_openai(prompt)
        return self._call_anthropic(prompt)

    def _call_anthropic(self, prompt: str) -> str | None:
        try:
            import anthropic
        except ImportError:
            return None
        client = anthropic.Anthropic(api_key=self.judge_api_key)
        for attempt in range(MAX_RETRIES + 1):
            try:
                msg = client.messages.create(
                    model=self.judge_model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S * (attempt + 1))
                    continue
                return None
        return None

    def _call_openai(self, prompt: str) -> str | None:
        try:
            import openai
            import os
            # Groq uses OpenAI-compatible API
            if "llama" in self.judge_model or "mixtral" in self.judge_model or "gemma" in self.judge_model:
                client = openai.OpenAI(
                    api_key=self.judge_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            else:
                client = openai.OpenAI(api_key=self.judge_api_key)
        except Exception:
            return None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.judge_model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S * (attempt + 1))
                    continue
                return None
        return None

    @staticmethod
    def _parse_judge_response(response_text: str) -> float | None:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except (json.JSONDecodeError, ValueError):
                    return None
            else:
                return None
        try:
            dims = [
                max(0.0, min(1.0, float(parsed.get(k, 0.0))))
                for k in ("relevance", "coherence", "uncertainty")
            ]
            return float(sum(dims) / len(dims))
        except (TypeError, ValueError):
            return None