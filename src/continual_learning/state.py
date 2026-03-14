from __future__ import annotations

import random

from continual_learning.constants import (
    DEFAULT_INPUT_CHUNK_TOKEN_COUNT,
    EMPTY_SIGNAL,
    RANDOM_STATE_TOKEN_COUNT,
    RANDOM_STATE_TOKEN_POOL,
)


def build_random_state_text(*, generator: random.Random) -> str:
    tokens = generator.sample(RANDOM_STATE_TOKEN_POOL, RANDOM_STATE_TOKEN_COUNT)
    return " ".join(tokens)


def split_signal_tokens(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(token for token in text.split() if token)


def join_signal_tokens(tokens: tuple[str, ...]) -> str:
    if not tokens:
        return EMPTY_SIGNAL
    return " ".join(tokens)

def chunk_input_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_INPUT_CHUNK_TOKEN_COUNT,
) -> tuple[str, ...]:
    tokens = split_signal_tokens(text)
    if not tokens:
        return (EMPTY_SIGNAL,)
    return tuple(
        join_signal_tokens(tokens[index : index + chunk_size])
        for index in range(0, len(tokens), chunk_size)
    )


def render_signal(text: str) -> str:
    return text if text else "<empty>"
