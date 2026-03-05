from __future__ import annotations

import re

from continual_learning.constants import EMPTY_SIGNAL

PROTOCOL_CONTROL_TOKENS = frozenset({
    EMPTY_SIGNAL.lower(),
    "evaluate",
    "none",
    "unknown",
})


def unique_sorted_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({token for token in tokens if token}))


def tokens_from_text(text: str) -> tuple[str, ...]:
    raw_tokens = tuple(
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in PROTOCOL_CONTROL_TOKENS
    )
    return unique_sorted_tokens(raw_tokens)


def tokens_from_pattern(pattern: str) -> tuple[str, ...]:
    if pattern.lower() in PROTOCOL_CONTROL_TOKENS or pattern == "":
        return ()
    return unique_sorted_tokens(tuple(pattern.split("+")))


def build_pattern(tokens: tuple[str, ...]) -> str:
    return "+".join(tokens) if tokens else EMPTY_SIGNAL


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
