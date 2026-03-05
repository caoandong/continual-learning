from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from continual_learning.constants import (
    MAX_MEMORY_PATTERNS,
    MAX_PATTERN_TOKENS,
    MAX_PROTOCOL_TOKENS,
    MEMORY_MATCH_THRESHOLD,
    MEMORY_MERGE_THRESHOLD,
    NEURON_STATE_VERSION,
    RANDOM_STATE_TOKEN_COUNT,
    RANDOM_STATE_TOKEN_POOL,
)
from continual_learning.protocol import (
    build_pattern,
    extract_target_label,
    overlap_score,
    tokens_from_text,
    unique_sorted_tokens,
)

logger = logging.getLogger(__name__)


class MemoryEntryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: tuple[str, ...] = ()
    output_text: str = ""
    weight: int = 1


class StructuredNeuronStateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = NEURON_STATE_VERSION
    summary_tokens: tuple[str, ...] = ()
    memories: tuple[MemoryEntryModel, ...] = ()


class StructuredNeuronResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    new_state: StructuredNeuronStateModel
    activation_up: str = "unknown"
    feedback_down: str = "none"


@dataclass(frozen=True)
class MemoryEntry:
    input_tokens: tuple[str, ...]
    output_text: str
    output_tokens: tuple[str, ...]
    weight: int


@dataclass(frozen=True)
class ParsedNeuronState:
    summary_tokens: tuple[str, ...]
    memories: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class StateEvolutionInput:
    previous_state_text: str
    bottom_up: str
    top_down: str
    candidate_texts: tuple[str, ...] = ()
    preferred_activation: str = ""
    preferred_feedback: str = ""


def trim_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    unique_tokens = unique_sorted_tokens(tokens)
    return unique_tokens[:MAX_PATTERN_TOKENS]


def trim_protocol_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    unique_tokens = unique_sorted_tokens(tokens)
    return unique_tokens[:MAX_PROTOCOL_TOKENS]


def normalized_token_items(items: tuple[str, ...]) -> tuple[str, ...]:
    flattened = tuple(
        token
        for item in items
        for token in tokens_from_text(item)
    )
    return trim_tokens(flattened)


def build_output_tokens(text: str) -> tuple[str, ...]:
    return trim_tokens(tokens_from_text(text))


def memory_entry_from_tokens(
    input_tokens: tuple[str, ...],
    output_text: str,
    weight: int,
) -> MemoryEntry | None:
    trimmed_input_tokens = normalized_token_items(input_tokens)
    normalized_output_text = output_text.strip()
    output_tokens = build_output_tokens(normalized_output_text)
    if not trimmed_input_tokens:
        return None
    return MemoryEntry(
        input_tokens=trimmed_input_tokens,
        output_text=normalized_output_text,
        output_tokens=output_tokens,
        weight=max(weight, 1),
    )


def serialize_state(state: ParsedNeuronState) -> str:
    payload = StructuredNeuronStateModel(
        version=NEURON_STATE_VERSION,
        summary_tokens=state.summary_tokens,
        memories=tuple(
            MemoryEntryModel(
                input_tokens=item.input_tokens,
                output_text=item.output_text,
                weight=item.weight,
            )
            for item in state.memories
        ),
    )
    return payload.model_dump_json()


def empty_state() -> ParsedNeuronState:
    return ParsedNeuronState(summary_tokens=(), memories=())


def parsed_state_from_model(payload: StructuredNeuronStateModel) -> ParsedNeuronState:
    memories = tuple(
        entry
        for entry in (
            memory_entry_from_tokens(item.input_tokens, item.output_text, item.weight)
            for item in payload.memories
        )
        if entry is not None
    )
    return ParsedNeuronState(
        summary_tokens=normalized_token_items(payload.summary_tokens),
        memories=memories,
    )


def parse_state_text(state_text: str) -> ParsedNeuronState:
    if not state_text.strip():
        return empty_state()
    try:
        payload = StructuredNeuronStateModel.model_validate_json(state_text)
    except ValidationError:
        logger.debug("[state] Invalid state payload, resetting to empty state: %s", state_text)
        return empty_state()
    return parsed_state_from_model(payload)


def build_random_state_text(*, generator: random.Random) -> str:
    summary_tokens = tuple(
        sorted(generator.sample(RANDOM_STATE_TOKEN_POOL, k=RANDOM_STATE_TOKEN_COUNT)),
    )
    seed_memory = memory_entry_from_tokens(summary_tokens, "", 1)
    state = ParsedNeuronState(
        summary_tokens=summary_tokens,
        memories=(seed_memory,) if seed_memory is not None else (),
    )
    return serialize_state(state)


def merged_tokens(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    overlap = tuple(token for token in left if token in set(right))
    if overlap:
        return trim_tokens(overlap)
    return trim_tokens(left + right)


def same_memory_output(left: MemoryEntry, right: MemoryEntry) -> bool:
    if not left.output_tokens and not right.output_tokens:
        return True
    return left.output_tokens == right.output_tokens


def merge_memory_entries(left: MemoryEntry, right: MemoryEntry) -> MemoryEntry:
    merged_input_tokens = merged_tokens(left.input_tokens, right.input_tokens)
    preferred_output_text = left.output_text if left.output_text else right.output_text
    merged = memory_entry_from_tokens(
        merged_input_tokens,
        preferred_output_text,
        left.weight + right.weight,
    )
    if merged is None:
        raise ValueError("Merged memory entry unexpectedly empty")
    return merged


def compress_memories(memories: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
    merged: list[MemoryEntry] = []
    sorted_memories = sorted(
        memories,
        key=lambda item: (-item.weight, item.output_text, item.input_tokens),
    )
    for memory in sorted_memories:
        best_index = None
        best_score = 0.0
        for index, candidate in enumerate(merged):
            if not same_memory_output(memory, candidate):
                continue
            score = overlap_score(memory.input_tokens, candidate.input_tokens)
            if score < MEMORY_MERGE_THRESHOLD or score <= best_score:
                continue
            best_index = index
            best_score = score
        if best_index is None:
            merged.append(memory)
            continue
        merged[best_index] = merge_memory_entries(merged[best_index], memory)
    return tuple(
        sorted(
            merged,
            key=lambda item: (-item.weight, item.output_text, item.input_tokens),
        )[:MAX_MEMORY_PATTERNS]
    )


def unsupervised_memory_entry(memory: MemoryEntry) -> MemoryEntry | None:
    return memory_entry_from_tokens(memory.input_tokens, "", memory.weight)


def recompute_summary(state: ParsedNeuronState) -> tuple[str, ...]:
    counter: Counter[str] = Counter(state.summary_tokens)
    for memory in state.memories:
        for token in memory.input_tokens:
            counter[token] += memory.weight
        for token in memory.output_tokens:
            counter[token] += memory.weight
    if not counter:
        return ()
    top_tokens = tuple(token for token, _ in counter.most_common(RANDOM_STATE_TOKEN_COUNT))
    return trim_tokens(top_tokens)


def sanitize_state_update(
    *,
    previous_state: ParsedNeuronState,
    proposed_state: ParsedNeuronState,
    top_down: str,
) -> ParsedNeuronState:
    if extract_target_label(top_down) is not None:
        return proposed_state
    unlabeled_memories = tuple(
        entry
        for entry in (
            unsupervised_memory_entry(memory)
            for memory in proposed_state.memories
        )
        if entry is not None
    )
    preserved_labeled = tuple(
        memory for memory in previous_state.memories if memory.output_text
    )
    memories = compress_memories(unlabeled_memories + preserved_labeled)
    next_state = ParsedNeuronState(
        summary_tokens=proposed_state.summary_tokens,
        memories=memories,
    )
    return ParsedNeuronState(
        summary_tokens=recompute_summary(next_state),
        memories=memories,
    )


def add_memory(
    memories: tuple[MemoryEntry, ...],
    input_text: str,
    output_text: str = "",
) -> tuple[MemoryEntry, ...]:
    candidate = memory_entry_from_tokens(tokens_from_text(input_text), output_text, 1)
    if candidate is None:
        return memories
    return compress_memories(memories + (candidate,))


def strongest_output_match(
    state: ParsedNeuronState,
    input_tokens: tuple[str, ...],
) -> str | None:
    best_output_text = None
    best_output_tokens: tuple[str, ...] = ()
    best_score = 0.0
    ambiguous = False
    for memory in state.memories:
        if not memory.output_tokens:
            continue
        score = overlap_score(input_tokens, memory.input_tokens)
        if score < MEMORY_MATCH_THRESHOLD:
            continue
        weighted_score = score + (0.01 * memory.weight)
        if weighted_score > best_score:
            best_output_text = memory.output_text
            best_output_tokens = memory.output_tokens
            best_score = weighted_score
            ambiguous = False
            continue
        if weighted_score == best_score and memory.output_tokens != best_output_tokens:
            ambiguous = True
    return None if ambiguous else best_output_text


def projected_tokens(state: ParsedNeuronState, input_tokens: tuple[str, ...]) -> tuple[str, ...]:
    if not input_tokens:
        return ()
    counter: Counter[str] = Counter(input_tokens)
    input_token_set = set(input_tokens)
    for memory in state.memories:
        overlap = tuple(token for token in memory.input_tokens if token in input_token_set)
        for token in overlap:
            counter[token] += memory.weight
    for token in state.summary_tokens:
        if token in input_token_set:
            counter[token] += 1
    top_tokens = tuple(token for token, _ in counter.most_common(MAX_PROTOCOL_TOKENS))
    return trim_protocol_tokens(top_tokens)


def target_feedback_tokens(
    state: ParsedNeuronState,
    output_text: str,
    input_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    target_tokens = build_output_tokens(output_text)
    matching = tuple(memory for memory in state.memories if memory.output_tokens == target_tokens)
    if not matching:
        return trim_protocol_tokens(input_tokens)
    strongest = max(matching, key=lambda item: (item.weight, len(item.input_tokens)))
    return trim_protocol_tokens(strongest.input_tokens)


def evolve_state(input_data: StateEvolutionInput) -> ParsedNeuronState:
    state = parse_state_text(input_data.previous_state_text)
    memories = state.memories
    memories = add_memory(memories, input_data.bottom_up)
    if extract_target_label(input_data.top_down) is None:
        memories = add_memory(memories, input_data.top_down)
    for text in input_data.candidate_texts:
        memories = add_memory(memories, text)

    target_output_text = extract_target_label(input_data.top_down)
    if target_output_text is not None:
        memories = add_memory(memories, input_data.bottom_up, target_output_text)

    next_state = ParsedNeuronState(
        summary_tokens=state.summary_tokens,
        memories=memories,
    )
    return ParsedNeuronState(
        summary_tokens=recompute_summary(next_state),
        memories=memories,
    )


def choose_activation(input_data: StateEvolutionInput, state: ParsedNeuronState) -> str:
    target_output_text = extract_target_label(input_data.top_down)
    if target_output_text is not None:
        return target_output_text

    preferred_output = strongest_output_match(
        state,
        tokens_from_text(input_data.preferred_activation),
    )
    if preferred_output is not None:
        return preferred_output

    input_tokens = tokens_from_text(input_data.bottom_up)
    matched_output = strongest_output_match(state, input_tokens)
    if matched_output is not None:
        return matched_output

    projected = projected_tokens(state, input_tokens)
    if projected:
        return build_pattern(projected)

    preferred_tokens = trim_protocol_tokens(tokens_from_text(input_data.preferred_activation))
    if preferred_tokens:
        return build_pattern(preferred_tokens)
    return "unknown"


def choose_feedback(input_data: StateEvolutionInput, state: ParsedNeuronState) -> str:
    target_output_text = extract_target_label(input_data.top_down)
    input_tokens = tokens_from_text(input_data.bottom_up)
    if target_output_text is not None:
        return build_pattern(target_feedback_tokens(state, target_output_text, input_tokens))

    preferred_tokens = trim_protocol_tokens(tokens_from_text(input_data.preferred_feedback))
    if preferred_tokens:
        return build_pattern(preferred_tokens)

    projected = projected_tokens(state, input_tokens)
    return build_pattern(projected)


def state_response(input_data: StateEvolutionInput) -> tuple[str, str, str]:
    next_state = evolve_state(input_data)
    return (
        serialize_state(next_state),
        choose_activation(input_data, next_state),
        choose_feedback(input_data, next_state),
    )


def readout_response(
    *,
    state: ParsedNeuronState,
    bottom_up: str,
    top_down: str,
    preferred_activation: str = "",
    preferred_feedback: str = "",
) -> tuple[str, str]:
    input_data = StateEvolutionInput(
        previous_state_text=serialize_state(state),
        bottom_up=bottom_up,
        top_down=top_down,
        preferred_activation=preferred_activation,
        preferred_feedback=preferred_feedback,
    )
    return (
        choose_activation(input_data, state),
        choose_feedback(input_data, state),
    )


def debug_state_summary(state_text: str) -> str:
    state = parse_state_text(state_text)
    return (
        f"summary={build_pattern(state.summary_tokens)} "
        f"memories={[(build_pattern(item.input_tokens), item.output_text, item.weight) for item in state.memories]}"
    )
