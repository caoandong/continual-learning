from __future__ import annotations

import re

from continual_learning.constants import EMPTY_SIGNAL

PROTOCOL_CONTROL_TOKENS = frozenset({
    EMPTY_SIGNAL.lower(),
    "evaluate",
    "none",
    "unknown",
})


def unique_preserving_order(tokens: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return tuple(result)


def split_atomic_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in re.split(r"[\s+]+", text.strip())
        if token
    )


def tokens_from_text(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in split_atomic_tokens(text)
        if token not in PROTOCOL_CONTROL_TOKENS
    )


def tokens_from_pattern(pattern: str) -> tuple[str, ...]:
    if pattern.lower() in PROTOCOL_CONTROL_TOKENS or pattern == "":
        return ()
    return tuple(
        token
        for token in split_atomic_tokens(pattern)
        if token not in PROTOCOL_CONTROL_TOKENS
    )


def build_pattern(tokens: tuple[str, ...]) -> str:
    compact = unique_preserving_order(tokens)
    return "+".join(compact) if compact else EMPTY_SIGNAL


def canonicalize_signal(text: str) -> str:
    return build_pattern(tokens_from_text(text))


def overlap_score(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    overlap = len(left_set & right_set)
    union = len(left_set | right_set)
    return overlap / union if union else 0.0


def extract_target_label(feedback: str) -> str | None:
    match = re.search(r"Target:\s*(.+)$", feedback)
    return match.group(1).strip() if match else None


def extract_predicted_label(feedback: str) -> str | None:
    match = re.search(r"Predicted:\s*([^|]+)", feedback)
    return match.group(1).strip() if match else None
