from __future__ import annotations

from typing import Any, Sequence


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def compute_answer_format_reward(prediction_text: str, prediction_lines: Sequence[str]) -> dict:
    """Reward KBQA answer output format from the remote answer model (not retriever actions)."""
    text = prediction_text or ""
    lines = [safe_text(line) for line in (prediction_lines or []) if safe_text(line)]
    issues: list[str] = []

    if not safe_text(text):
        issues.append("empty_answer")
        return {"format_reward": 0.0, "format_issues": issues}

    if "<think>" in text or "```" in text:
        issues.append("forbidden_wrapper")
        return {"format_reward": 0.0, "format_issues": issues}

    lower = text.lower()
    if any(marker in lower for marker in ["because", "therefore", "triples:", "reasoning", "analysis:"]):
        issues.append("verbose_reasoning")
        reward = 0.2
    elif len(lines) == 1:
        reward = 1.0
    elif len(lines) <= 5:
        reward = 0.8
    else:
        issues.append("too_many_answer_lines")
        reward = 0.4

    return {"format_reward": reward, "format_issues": issues}
