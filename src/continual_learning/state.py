from __future__ import annotations
import logging
import random
from collections import Counter
from dataclasses import dataclass
from math import log
from math import ceil
from math import sqrt

from pydantic import BaseModel, ConfigDict, ValidationError

from continual_learning.constants import (
    COMPETING_MEMORY_WEAKEN_THRESHOLD,
    FEEDBACK_CHANNEL_MATCH_WEIGHT,
    GENERIC_SHINGLE_WIDTHS,
    LONG_STREAM_TOKEN_THRESHOLD,
    MAX_ANCHOR_SIGNATURE_TOKENS,
    MAX_DELTA_WINDOW_SIGNATURE_TOKENS,
    MAX_FEEDBACK_RESIDUAL_TOKENS,
    MAX_GLOBAL_SIGNATURE_TOKENS,
    MAX_MEMORY_PATTERNS,
    MAX_MEMORY_INPUT_TOKENS,
    MAX_NUMERIC_RELATION_SIGNATURE_TOKENS,
    NUMERIC_FINGERPRINT_MATCH_WEIGHT,
    NUMERIC_FINGERPRINT_SCALE,
    NUMERIC_FINGERPRINT_SIZE,
    MAX_NUMERIC_SEGMENT_TOKENS_PER_SCALE,
    MAX_NUMERIC_WINDOW_SIGNATURE_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_PROTOCOL_TOKENS,
    MAX_RETRIEVED_MEMORIES,
    MAX_SENSORY_RESIDUAL_TOKENS,
    MAX_SHINGLE_SIGNATURE_TOKENS,
    MAX_SUPPORT_MEMORIES,
    MAX_SUMMARY_TOKENS,
    MAX_WINDOW_SIGNATURE_TOKENS,
    MEMORY_MATCH_MARGIN,
    MEMORY_MATCH_THRESHOLD,
    MEMORY_MERGE_THRESHOLD,
    NEURON_STATE_VERSION,
    NUMERIC_SEGMENT_SCALES,
    PROTOTYPE_MATCH_WEIGHT,
    RANDOM_STATE_TOKEN_COUNT,
    RANDOM_STATE_TOKEN_POOL,
    SENSORY_CHANNEL_MATCH_WEIGHT,
    STREAM_ANCHOR_COUNT,
    STREAM_WINDOW_COUNT,
    STREAM_WINDOW_SIZE,
    UNSUPERVISED_MEMORY_MERGE_THRESHOLD,
    WEAK_MEMORY_MATCH_THRESHOLD,
)
from continual_learning.protocol import (
    build_pattern,
    extract_predicted_label,
    extract_target_label,
    overlap_score,
    tokens_from_text,
    tokens_from_pattern,
    unique_preserving_order,
)

logger = logging.getLogger(__name__)


class MemoryEntryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: tuple[str, ...] = ()
    numeric_fingerprint: tuple[int, ...] = ()
    output_text: str = ""
    weight: int = 1


class SupportMemoryModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    numeric_fingerprint: tuple[int, ...] = ()
    output_text: str = ""
    weight: int = 1


class StructuredNeuronStateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = NEURON_STATE_VERSION
    summary_tokens: tuple[str, ...] = ()
    memories: tuple[MemoryEntryModel, ...] = ()
    support_memories: tuple[SupportMemoryModel, ...] = ()


class StructuredNeuronResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    new_state: StructuredNeuronStateModel
    activation_up: str = "unknown"
    feedback_down: str = "none"


@dataclass(frozen=True)
class MemoryEntry:
    input_tokens: tuple[str, ...]
    numeric_fingerprint: tuple[int, ...]
    output_text: str
    output_tokens: tuple[str, ...]
    weight: int


@dataclass(frozen=True)
class SupportMemory:
    numeric_fingerprint: tuple[int, ...]
    output_text: str
    weight: int


@dataclass(frozen=True)
class ParsedNeuronState:
    summary_tokens: tuple[str, ...]
    memories: tuple[MemoryEntry, ...]
    support_memories: tuple[SupportMemory, ...]


@dataclass(frozen=True)
class StateEvolutionInput:
    previous_state_text: str
    bottom_up: str
    top_down: str
    sensory_input: str = ""
    candidate_texts: tuple[str, ...] = ()
    preferred_activation: str = ""
    preferred_feedback: str = ""


def trim_memory_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return unique_preserving_order(tokens)[:MAX_MEMORY_INPUT_TOKENS]


def trim_protocol_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return unique_preserving_order(tokens)[:MAX_PROTOCOL_TOKENS]


def trim_output_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return unique_preserving_order(tokens)[:MAX_OUTPUT_TOKENS]


def trim_summary_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return unique_preserving_order(tokens)[:MAX_SUMMARY_TOKENS]


def trim_signature_tokens(tokens: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    return unique_preserving_order(tokens)[:limit]


def anchor_indices(*, count: int, size: int) -> tuple[int, ...]:
    if count <= 0 or size <= 0:
        return ()
    if size <= count:
        return tuple(range(size))
    last_index = size - 1
    indices: list[int] = []
    seen: set[int] = set()
    for index in range(count):
        anchor = round((last_index * index) / (count - 1))
        if anchor in seen:
            continue
        seen.add(anchor)
        indices.append(anchor)
    return tuple(indices)


def dominant_stream_token(tokens: tuple[str, ...]) -> str:
    if not tokens:
        return ""
    return Counter(tokens).most_common(1)[0][0]


def window_score(*, dominant_token: str, start: int, tokens: tuple[str, ...]) -> tuple[int, int, int]:
    window = tokens[start:start + STREAM_WINDOW_SIZE]
    non_dominant = sum(1 for token in window if token != dominant_token)
    diversity = len(set(window))
    transitions = sum(1 for left, right in zip(window, window[1:]) if left != right)
    return non_dominant, diversity, transitions


def informative_window_starts(tokens: tuple[str, ...]) -> tuple[int, ...]:
    if not tokens:
        return ()
    if len(tokens) <= STREAM_WINDOW_SIZE:
        return (0,)
    dominant_token = dominant_stream_token(tokens)
    starts = tuple(range(0, len(tokens) - STREAM_WINDOW_SIZE + 1, STREAM_WINDOW_SIZE))
    ranked = sorted(
        starts,
        key=lambda start: window_score(dominant_token=dominant_token, start=start, tokens=tokens),
        reverse=True,
    )
    return tuple(sorted(ranked[:STREAM_WINDOW_COUNT]))


def numeric_values(tokens: tuple[str, ...]) -> tuple[int, ...] | None:
    values: list[int] = []
    for token in tokens:
        if not token.lstrip("-").isdigit():
            return None
        values.append(int(token))
    return tuple(values)


def pooled_numeric_values(
    values: tuple[int, ...],
    *,
    size: int,
) -> tuple[float, ...]:
    if not values or size <= 0:
        return ()
    pooled: list[float] = []
    for index in range(size):
        start = (len(values) * index) // size
        end = (len(values) * (index + 1)) // size
        chunk = values[start:end]
        if not chunk:
            pooled.append(0.0)
            continue
        pooled.append(sum(chunk) / len(chunk))
    return tuple(pooled)


def normalized_numeric_fingerprint(values: tuple[float, ...]) -> tuple[int, ...]:
    if not values:
        return ()
    magnitude = sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        return tuple(0 for _ in values)
    return tuple(
        round((value / magnitude) * NUMERIC_FINGERPRINT_SCALE)
        for value in values
    )


def trim_numeric_fingerprint(
    fingerprint: tuple[int, ...],
) -> tuple[int, ...]:
    return fingerprint[:NUMERIC_FINGERPRINT_SIZE]


def numeric_fingerprint_from_tokens(tokens: tuple[str, ...]) -> tuple[int, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    pooled = pooled_numeric_values(values, size=NUMERIC_FINGERPRINT_SIZE)
    return trim_numeric_fingerprint(normalized_numeric_fingerprint(pooled))


def numeric_fingerprint_from_text(text: str) -> tuple[int, ...]:
    return numeric_fingerprint_from_tokens(tokens_from_text(text))


def magnitude_bucket(value: int) -> str:
    magnitude = abs(value)
    if magnitude == 0:
        return "z"
    if magnitude < 32:
        return "b1"
    if magnitude < 96:
        return "b2"
    if magnitude < 160:
        return "b3"
    if magnitude < 224:
        return "b4"
    return "b5"


def fraction_bucket(*, numerator: int, denominator: int) -> str:
    if denominator <= 0 or numerator <= 0:
        return "0"
    ratio = numerator / denominator
    if ratio < 0.2:
        return "1"
    if ratio < 0.4:
        return "2"
    if ratio < 0.6:
        return "3"
    if ratio < 0.8:
        return "4"
    return "5"


def motif_token(values: tuple[int, ...]) -> str:
    indices = anchor_indices(count=4, size=len(values))
    sampled = tuple(magnitude_bucket(values[index]) for index in indices)
    return f"m:{'_'.join(sampled)}"


def summary_token(*, dominant_value: int, values: tuple[int, ...]) -> str:
    transitions = sum(1 for left, right in zip(values, values[1:]) if left != right)
    non_dominant = sum(1 for value in values if value != dominant_value)
    peak = max(values, key=abs, default=0)
    return (
        "s:"
        f"nz{fraction_bucket(numerator=non_dominant, denominator=len(values))}:"
        f"tr{fraction_bucket(numerator=transitions, denominator=max(len(values) - 1, 1))}:"
        f"mx{magnitude_bucket(peak)}"
    )


def delta_values(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        right - left
        for left, right in zip(values, values[1:])
    )


def delta_bucket(value: int) -> str:
    if value == 0:
        return "z"
    direction = "p" if value > 0 else "n"
    return f"{direction}{magnitude_bucket(value)}"


def delta_motif_token(values: tuple[int, ...]) -> str:
    deltas = delta_values(values)
    if not deltas:
        return "dm:z"
    indices = anchor_indices(count=4, size=len(deltas))
    sampled = tuple(delta_bucket(deltas[index]) for index in indices)
    return f"dm:{'_'.join(sampled)}"


def delta_summary_token(values: tuple[int, ...]) -> str:
    deltas = delta_values(values)
    if not deltas:
        return "ds:up0:dn0:mxz"
    positive = sum(1 for delta in deltas if delta > 0)
    negative = sum(1 for delta in deltas if delta < 0)
    peak = max(deltas, key=abs, default=0)
    return (
        "ds:"
        f"up{fraction_bucket(numerator=positive, denominator=len(deltas))}:"
        f"dn{fraction_bucket(numerator=negative, denominator=len(deltas))}:"
        f"mx{delta_bucket(peak)}"
    )


def numeric_window_signature_token(
    *,
    dominant_value: int,
    values: tuple[int, ...],
    start: int,
) -> str:
    return (
        f"wv{start}:"
        f"{motif_token(values)}:"
        f"{summary_token(dominant_value=dominant_value, values=values)}"
    )


def numeric_window_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    dominant_value = Counter(values).most_common(1)[0][0]
    starts = informative_window_starts(tokens)
    return tuple(
        numeric_window_signature_token(
            dominant_value=dominant_value,
            values=values[start:start + STREAM_WINDOW_SIZE],
            start=start,
        )
        for start in starts
    )


def global_numeric_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    transitions = sum(1 for left, right in zip(values, values[1:]) if left != right)
    non_zero = sum(1 for value in values if value != 0)
    peak = max(values, key=abs, default=0)
    return (
        f"g:nz{fraction_bucket(numerator=non_zero, denominator=len(values))}",
        f"g:tr{fraction_bucket(numerator=transitions, denominator=max(len(values) - 1, 1))}",
        f"g:mx{magnitude_bucket(peak)}",
    )


def global_delta_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    deltas = delta_values(values)
    if not deltas:
        return ()
    positive = sum(1 for delta in deltas if delta > 0)
    negative = sum(1 for delta in deltas if delta < 0)
    peak = max(deltas, key=abs, default=0)
    return (
        f"dg:up{fraction_bucket(numerator=positive, denominator=len(deltas))}",
        f"dg:dn{fraction_bucket(numerator=negative, denominator=len(deltas))}",
        f"dg:mx{delta_bucket(peak)}",
    )


def delta_window_signature_token(values: tuple[int, ...], *, start: int) -> str:
    return f"wd{start}:{delta_motif_token(values)}:{delta_summary_token(values)}"


def delta_window_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    starts = informative_window_starts(tokens)
    return tuple(
        delta_window_signature_token(
            values[start:start + STREAM_WINDOW_SIZE],
            start=start,
        )
        for start in starts
    )


def segment_bounds(*, size: int, count: int, index: int) -> tuple[int, int]:
    start = (size * index) // count
    end = (size * (index + 1)) // count
    return start, end


def segment_values(values: tuple[int, ...], *, count: int, index: int) -> tuple[int, ...]:
    start, end = segment_bounds(size=len(values), count=count, index=index)
    return values[start:end]


def segment_centroid_bucket(values: tuple[int, ...]) -> str:
    total = sum(values)
    if total <= 0:
        return "0"
    weighted_center = sum((index + 1) * value for index, value in enumerate(values))
    normalized = (weighted_center / total) / len(values)
    if normalized < 0.2:
        return "1"
    if normalized < 0.4:
        return "2"
    if normalized < 0.6:
        return "3"
    if normalized < 0.8:
        return "4"
    return "5"


def segment_summary_token(
    values: tuple[int, ...],
    *,
    scale: int,
    index: int,
) -> str:
    if not values:
        return ""
    non_zero = sum(1 for value in values if value != 0)
    high_magnitude = sum(1 for value in values if abs(value) >= 192)
    peak = max(values, key=abs, default=0)
    return (
        f"s{scale}:{index}:"
        f"nz{fraction_bucket(numerator=non_zero, denominator=len(values))}:"
        f"hi{fraction_bucket(numerator=high_magnitude, denominator=len(values))}:"
        f"pk{magnitude_bucket(peak)}:"
        f"ct{segment_centroid_bucket(values)}"
    )


def numeric_segment_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    return tuple(
        token
        for scale in NUMERIC_SEGMENT_SCALES
        for index in range(scale)
        if (
            token := segment_summary_token(
                segment_values(values, count=scale, index=index),
                scale=scale,
                index=index,
            )
        )
    )


def sampled_segment_indices(*, scale: int) -> tuple[int, ...]:
    return sample_positions(
        positions=tuple(range(scale)),
        count=MAX_NUMERIC_SEGMENT_TOKENS_PER_SCALE,
    )


def numeric_segment_signature_tokens_for_scale(tokens: tuple[str, ...], *, scale: int) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    return tuple(
        token
        for index in sampled_segment_indices(scale=scale)
        if (
            token := segment_summary_token(
                segment_values(values, count=scale, index=index),
                scale=scale,
                index=index,
            )
        )
    )


def change_bucket(*, left: int, right: int) -> str:
    denominator = max(abs(left), abs(right), 1)
    delta = (right - left) / denominator
    if delta <= -0.6:
        return "nn"
    if delta <= -0.2:
        return "n"
    if delta < 0.2:
        return "z"
    if delta < 0.6:
        return "p"
    return "pp"


def segment_mass(values: tuple[int, ...]) -> int:
    return sum(abs(value) for value in values)


def segment_non_zero_count(values: tuple[int, ...]) -> int:
    return sum(1 for value in values if value != 0)


def relation_token(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    scale: int,
    index: int,
) -> str:
    return (
        f"r{scale}:{index}:"
        f"nz{change_bucket(left=segment_non_zero_count(left), right=segment_non_zero_count(right))}:"
        f"ms{change_bucket(left=segment_mass(left), right=segment_mass(right))}"
    )


def numeric_relation_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    values = numeric_values(tokens)
    if values is None:
        return ()
    return tuple(
        relation_token(
            segment_values(values, count=scale, index=index),
            segment_values(values, count=scale, index=index + 1),
            scale=scale,
            index=index,
        )
        for scale in NUMERIC_SEGMENT_SCALES
        for index in range(scale - 1)
    )


def sampled_relation_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    relation_tokens = numeric_relation_signature_tokens(tokens)
    if len(relation_tokens) <= MAX_NUMERIC_RELATION_SIGNATURE_TOKENS:
        return relation_tokens
    sampled_indices = sample_positions(
        positions=tuple(range(len(relation_tokens))),
        count=MAX_NUMERIC_RELATION_SIGNATURE_TOKENS,
    )
    return tuple(relation_tokens[index] for index in sampled_indices)


def sampled_shingle_starts(tokens: tuple[str, ...], *, width: int) -> tuple[int, ...]:
    if len(tokens) < width:
        return ()
    starts = informative_window_starts(tokens)
    if not starts:
        return (0,)
    return tuple(
        start
        for start in starts
        if start + width <= len(tokens)
    )


def shingle_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"n{width}:{start}:{'_'.join(tokens[start:start + width])}"
        for width in GENERIC_SHINGLE_WIDTHS
        for start in sampled_shingle_starts(tokens, width=width)
    )


def sample_positions(*, positions: tuple[int, ...], count: int) -> tuple[int, ...]:
    if len(positions) <= count:
        return positions
    sampled_indices = anchor_indices(count=count, size=len(positions))
    return tuple(positions[index] for index in sampled_indices)


def informative_anchor_positions(tokens: tuple[str, ...]) -> tuple[int, ...]:
    dominant_token = dominant_stream_token(tokens)
    candidate_positions = tuple(
        start + offset
        for start in informative_window_starts(tokens)
        for offset, token in enumerate(tokens[start:start + STREAM_WINDOW_SIZE])
        if token != dominant_token
    )
    if candidate_positions:
        return sample_positions(positions=candidate_positions, count=STREAM_ANCHOR_COUNT)
    return sample_positions(
        positions=tuple(range(len(tokens))),
        count=STREAM_ANCHOR_COUNT,
    )


def anchor_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"a{index}:{tokens[index]}"
        for index in informative_anchor_positions(tokens)
    )


def window_signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    starts = informative_window_starts(tokens)
    return tuple(
        f"w{start}:{'_'.join(tokens[start:start + STREAM_WINDOW_SIZE])}"
        for start in starts
    )


def signature_token_groups(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = [
        (f"len:{len(tokens)}",) + trim_signature_tokens(
            global_numeric_signature_tokens(tokens) + global_delta_signature_tokens(tokens),
            limit=MAX_GLOBAL_SIGNATURE_TOKENS,
        ),
    ]
    groups.extend(
        trim_signature_tokens(
            numeric_segment_signature_tokens_for_scale(tokens, scale=scale),
            limit=MAX_NUMERIC_SEGMENT_TOKENS_PER_SCALE,
        )
        for scale in NUMERIC_SEGMENT_SCALES
    )
    groups.append(
        trim_signature_tokens(
            sampled_relation_signature_tokens(tokens),
            limit=MAX_NUMERIC_RELATION_SIGNATURE_TOKENS,
        ),
    )
    groups.append(
        trim_signature_tokens(
            numeric_window_signature_tokens(tokens),
            limit=MAX_NUMERIC_WINDOW_SIGNATURE_TOKENS,
        ),
    )
    groups.append(
        trim_signature_tokens(
            delta_window_signature_tokens(tokens),
            limit=MAX_DELTA_WINDOW_SIGNATURE_TOKENS,
        ),
    )
    groups.append(
        trim_signature_tokens(
            shingle_signature_tokens(tokens),
            limit=MAX_SHINGLE_SIGNATURE_TOKENS,
        ),
    )
    groups.append(
        trim_signature_tokens(
            anchor_signature_tokens(tokens),
            limit=MAX_ANCHOR_SIGNATURE_TOKENS,
        ),
    )
    groups.append(
        trim_signature_tokens(
            window_signature_tokens(tokens),
            limit=MAX_WINDOW_SIGNATURE_TOKENS,
        ),
    )
    return tuple(group for group in groups if group)


def signature_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    if len(tokens) <= LONG_STREAM_TOKEN_THRESHOLD:
        return trim_memory_tokens(tokens)
    return interleave_token_groups(signature_token_groups(tokens))


def normalized_token_items(items: tuple[str, ...]) -> tuple[str, ...]:
    flattened = tuple(
        token
        for item in items
        for token in tokens_from_text(item)
    )
    return trim_memory_tokens(flattened)


def build_output_tokens(text: str) -> tuple[str, ...]:
    return trim_output_tokens(tokens_from_text(text))


def memory_tokens_from_text(text: str) -> tuple[str, ...]:
    return signature_tokens(tokens_from_text(text))


def merged_memory_tokens(texts: tuple[str, ...]) -> tuple[str, ...]:
    combined = tuple(
        token
        for text in texts
        for token in memory_tokens_from_text(text)
    )
    return trim_memory_tokens(combined)


def prefixed_tokens(prefix: str, tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}:{token}" for token in tokens)


def interleave_token_groups(groups: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    if not groups:
        return ()
    max_group_length = max(len(group) for group in groups)
    interleaved: list[str] = []
    seen: set[str] = set()
    for index in range(max_group_length):
        for group in groups:
            if index >= len(group):
                continue
            token = group[index]
            if token in seen:
                continue
            seen.add(token)
            interleaved.append(token)
            if len(interleaved) >= MAX_MEMORY_INPUT_TOKENS:
                return tuple(interleaved)
    return tuple(interleaved)


def channel_tokens(prefix: str, text: str) -> tuple[str, ...]:
    return prefixed_tokens(prefix, memory_tokens_from_text(text))


def shared_channel_tokens(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    right_set = set(right)
    return tuple(f"shared:{token}" for token in left if token in right_set)


def bottom_up_message_tokens(text: str) -> tuple[str, ...]:
    return trim_memory_tokens(tokens_from_pattern(text))


def memory_entry_from_tokens(
    input_tokens: tuple[str, ...],
    output_text: str,
    weight: int,
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> MemoryEntry | None:
    trimmed_input_tokens = trim_memory_tokens(input_tokens)
    normalized_output_text = output_text.strip()
    output_tokens = build_output_tokens(normalized_output_text)
    if not trimmed_input_tokens:
        return None
    return MemoryEntry(
        input_tokens=trimmed_input_tokens,
        numeric_fingerprint=trim_numeric_fingerprint(numeric_fingerprint),
        output_text=normalized_output_text,
        output_tokens=output_tokens,
        weight=max(weight, 1),
    )


def support_memory_from_fingerprint(
    numeric_fingerprint: tuple[int, ...],
    output_text: str,
    weight: int,
) -> SupportMemory | None:
    normalized_output_text = output_text.strip()
    trimmed_fingerprint = trim_numeric_fingerprint(numeric_fingerprint)
    if not trimmed_fingerprint or not normalized_output_text:
        return None
    return SupportMemory(
        numeric_fingerprint=trimmed_fingerprint,
        output_text=normalized_output_text,
        weight=max(weight, 1),
    )


def serialize_state(state: ParsedNeuronState) -> str:
    payload = StructuredNeuronStateModel(
        version=NEURON_STATE_VERSION,
        summary_tokens=state.summary_tokens,
        memories=tuple(
            MemoryEntryModel(
                input_tokens=item.input_tokens,
                numeric_fingerprint=item.numeric_fingerprint,
                output_text=item.output_text,
                weight=item.weight,
            )
            for item in state.memories
        ),
        support_memories=tuple(
            SupportMemoryModel(
                numeric_fingerprint=item.numeric_fingerprint,
                output_text=item.output_text,
                weight=item.weight,
            )
            for item in state.support_memories
        ),
    )
    return payload.model_dump_json()


def empty_state() -> ParsedNeuronState:
    return ParsedNeuronState(summary_tokens=(), memories=(), support_memories=())


def parsed_state_from_model(payload: StructuredNeuronStateModel) -> ParsedNeuronState:
    memories = tuple(
        entry
        for entry in (
            memory_entry_from_tokens(
                normalized_token_items(item.input_tokens),
                item.output_text,
                item.weight,
                numeric_fingerprint=trim_numeric_fingerprint(item.numeric_fingerprint),
            )
            for item in payload.memories
        )
        if entry is not None
    )
    support_memories = tuple(
        entry
        for entry in (
            support_memory_from_fingerprint(
                trim_numeric_fingerprint(item.numeric_fingerprint),
                item.output_text,
                item.weight,
            )
            for item in payload.support_memories
        )
        if entry is not None
    )
    return ParsedNeuronState(
        summary_tokens=trim_summary_tokens(normalized_token_items(payload.summary_tokens)),
        memories=memories,
        support_memories=support_memories,
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
    summary_tokens = trim_summary_tokens(
        tuple(sorted(generator.sample(RANDOM_STATE_TOKEN_POOL, k=RANDOM_STATE_TOKEN_COUNT))),
    )
    seed_memory = memory_entry_from_tokens(summary_tokens, "", 1)
    state = ParsedNeuronState(
        summary_tokens=summary_tokens,
        memories=(seed_memory,) if seed_memory is not None else (),
        support_memories=(),
    )
    return serialize_state(state)


def merged_tokens(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    overlap = tuple(token for token in left if token in set(right))
    if overlap:
        return trim_memory_tokens(overlap)
    return trim_memory_tokens(left + right)


def same_memory_output(left: MemoryEntry, right: MemoryEntry) -> bool:
    if not left.output_tokens and not right.output_tokens:
        return True
    return left.output_tokens == right.output_tokens


def merged_numeric_fingerprint(
    left: MemoryEntry,
    right: MemoryEntry,
) -> tuple[int, ...]:
    if not left.numeric_fingerprint:
        return right.numeric_fingerprint
    if not right.numeric_fingerprint:
        return left.numeric_fingerprint
    if len(left.numeric_fingerprint) != len(right.numeric_fingerprint):
        return left.numeric_fingerprint
    total_weight = left.weight + right.weight
    if total_weight <= 0:
        return left.numeric_fingerprint
    return tuple(
        round(
            ((left_value * left.weight) + (right_value * right.weight))
            / total_weight,
        )
        for left_value, right_value in zip(
            left.numeric_fingerprint,
            right.numeric_fingerprint,
        )
    )


def merge_memory_entries(left: MemoryEntry, right: MemoryEntry) -> MemoryEntry:
    merged_input_tokens = merged_tokens(left.input_tokens, right.input_tokens)
    preferred_output_text = left.output_text if left.output_text else right.output_text
    merged = memory_entry_from_tokens(
        merged_input_tokens,
        preferred_output_text,
        left.weight + right.weight,
        numeric_fingerprint=merged_numeric_fingerprint(left, right),
    )
    if merged is None:
        raise ValueError("Merged memory entry unexpectedly empty")
    return merged


def memory_sort_key(memory: MemoryEntry) -> tuple[int, str, tuple[str, ...]]:
    return (-memory.weight, memory.output_text, memory.input_tokens)


def sorted_memories(memories: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
    return tuple(sorted(memories, key=memory_sort_key))


def matching_memory_index(
    *,
    memory: MemoryEntry,
    merged: tuple[MemoryEntry, ...],
    merge_threshold: float,
) -> int | None:
    best_index = None
    best_score = 0.0
    for index, candidate in enumerate(merged):
        if not same_memory_output(memory, candidate):
            continue
        score = overlap_score(memory.input_tokens, candidate.input_tokens)
        if score < merge_threshold or score <= best_score:
            continue
        best_index = index
        best_score = score
    return best_index


def merge_compatible_memories(
    memories: tuple[MemoryEntry, ...],
    *,
    merge_threshold: float,
) -> tuple[MemoryEntry, ...]:
    merged: list[MemoryEntry] = []
    for memory in sorted_memories(memories):
        best_index = matching_memory_index(
            memory=memory,
            merged=tuple(merged),
            merge_threshold=merge_threshold,
        )
        if best_index is None:
            merged.append(memory)
            continue
        merged[best_index] = merge_memory_entries(merged[best_index], memory)
    return sorted_memories(tuple(merged))


def select_label_representatives(memories: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
    representatives: dict[str, MemoryEntry] = {}
    for memory in memories:
        if not memory.output_text or memory.output_text in representatives:
            continue
        representatives[memory.output_text] = memory
    return tuple(representatives.values())


def remove_selected_memories(
    *,
    memories: tuple[MemoryEntry, ...],
    selected: tuple[MemoryEntry, ...],
) -> tuple[MemoryEntry, ...]:
    selected_counts: Counter[MemoryEntry] = Counter(selected)
    remaining: list[MemoryEntry] = []
    for memory in memories:
        if selected_counts[memory] > 0:
            selected_counts[memory] -= 1
            continue
        remaining.append(memory)
    return tuple(remaining)


def compress_memories(memories: tuple[MemoryEntry, ...]) -> tuple[MemoryEntry, ...]:
    labeled = tuple(memory for memory in memories if memory.output_text)
    unlabeled = tuple(memory for memory in memories if not memory.output_text)
    merged_labeled = merge_compatible_memories(
        labeled,
        merge_threshold=MEMORY_MERGE_THRESHOLD,
    )
    merged_unlabeled = merge_compatible_memories(
        unlabeled,
        merge_threshold=UNSUPERVISED_MEMORY_MERGE_THRESHOLD,
    )
    label_representatives = select_label_representatives(merged_labeled)
    retained = sorted_memories(label_representatives)[:MAX_MEMORY_PATTERNS]
    remaining_slots = MAX_MEMORY_PATTERNS - len(retained)
    if remaining_slots <= 0:
        return retained
    extra_labeled = remove_selected_memories(
        memories=merged_labeled,
        selected=retained,
    )
    extras = sorted_memories(extra_labeled) + sorted_memories(merged_unlabeled)
    return sorted_memories(retained + extras[:remaining_slots])


def support_memory_sort_key(
    memory: SupportMemory,
) -> tuple[int, str, tuple[int, ...]]:
    return (-memory.weight, memory.output_text, memory.numeric_fingerprint)


def merge_support_memories(
    memories: tuple[SupportMemory, ...],
) -> tuple[SupportMemory, ...]:
    merged: dict[tuple[str, tuple[int, ...]], SupportMemory] = {}
    for memory in memories:
        key = (memory.output_text, memory.numeric_fingerprint)
        existing = merged.get(key)
        if existing is None:
            merged[key] = memory
            continue
        merged[key] = SupportMemory(
            numeric_fingerprint=memory.numeric_fingerprint,
            output_text=memory.output_text,
            weight=existing.weight + memory.weight,
        )
    return tuple(
        sorted(
            merged.values(),
            key=support_memory_sort_key,
        )[:MAX_SUPPORT_MEMORIES]
    )


def add_support_memory(
    support_memories: tuple[SupportMemory, ...],
    *,
    numeric_fingerprint: tuple[int, ...],
    output_text: str,
) -> tuple[SupportMemory, ...]:
    candidate = support_memory_from_fingerprint(
        numeric_fingerprint,
        output_text,
        1,
    )
    if candidate is None:
        return support_memories
    return merge_support_memories(support_memories + (candidate,))


def unsupervised_memory_entry(memory: MemoryEntry) -> MemoryEntry | None:
    return memory_entry_from_tokens(
        memory.input_tokens,
        "",
        memory.weight,
        numeric_fingerprint=memory.numeric_fingerprint,
    )


def recompute_summary(state: ParsedNeuronState) -> tuple[str, ...]:
    counter: Counter[str] = Counter(state.summary_tokens)
    for memory in state.memories:
        for token in memory.input_tokens:
            counter[token] += memory.weight
        for token in memory.output_tokens:
            counter[token] += memory.weight
    for support_memory in state.support_memories:
        for token in build_output_tokens(support_memory.output_text):
            counter[token] += support_memory.weight
    if not counter:
        return ()
    top_tokens = tuple(token for token, _ in counter.most_common(MAX_SUMMARY_TOKENS))
    return trim_summary_tokens(top_tokens)


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
        support_memories=previous_state.support_memories,
    )
    return ParsedNeuronState(
        summary_tokens=recompute_summary(next_state),
        memories=memories,
        support_memories=next_state.support_memories,
    )


def merge_state_updates(
    left: ParsedNeuronState,
    right: ParsedNeuronState,
) -> ParsedNeuronState:
    memories = compress_memories(left.memories + right.memories)
    support_memories = merge_support_memories(
        left.support_memories + right.support_memories,
    )
    next_state = ParsedNeuronState(
        summary_tokens=trim_summary_tokens(left.summary_tokens + right.summary_tokens),
        memories=memories,
        support_memories=support_memories,
    )
    return ParsedNeuronState(
        summary_tokens=recompute_summary(next_state),
        memories=memories,
        support_memories=support_memories,
    )


def add_memory(
    memories: tuple[MemoryEntry, ...],
    input_text: str,
    output_text: str = "",
) -> tuple[MemoryEntry, ...]:
    candidate = memory_entry_from_tokens(
        memory_tokens_from_text(input_text),
        output_text,
        1,
        numeric_fingerprint=numeric_fingerprint_from_text(input_text),
    )
    if candidate is None:
        return memories
    return compress_memories(memories + (candidate,))


def add_memory_tokens(
    memories: tuple[MemoryEntry, ...],
    input_tokens: tuple[str, ...],
    output_text: str = "",
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> tuple[MemoryEntry, ...]:
    candidate = memory_entry_from_tokens(
        input_tokens,
        output_text,
        1,
        numeric_fingerprint=numeric_fingerprint,
    )
    if candidate is None:
        return memories
    return compress_memories(memories + (candidate,))


def label_memory_groups(state: ParsedNeuronState) -> dict[str, tuple[MemoryEntry, ...]]:
    grouped: dict[str, list[MemoryEntry]] = {}
    for memory in state.memories:
        if not memory.output_text:
            continue
        grouped.setdefault(memory.output_text, []).append(memory)
    return {
        label: tuple(memories)
        for label, memories in grouped.items()
    }


def label_prototype_tokens(memories: tuple[MemoryEntry, ...]) -> tuple[str, ...]:
    if not memories:
        return ()
    document_frequency: Counter[str] = Counter()
    token_order: list[str] = []
    seen_order: set[str] = set()
    for memory in memories:
        document_frequency.update(set(memory.input_tokens))
        for token in memory.input_tokens:
            if token in seen_order:
                continue
            seen_order.add(token)
            token_order.append(token)
    minimum_support = max(1, ceil(len(memories) / 2))
    consensus = tuple(
        token
        for token in token_order
        if document_frequency[token] >= minimum_support
    )
    order_index = {token: index for index, token in enumerate(token_order)}
    ranked = tuple(
        sorted(
            token_order,
            key=lambda token: (-document_frequency[token], order_index[token]),
        ),
    )
    if not consensus:
        return trim_memory_tokens(ranked)
    remaining = tuple(
        token
        for token in ranked
        if token not in set(consensus)
    )
    return trim_memory_tokens(consensus + remaining)


def token_specificity_weights(
    grouped_memories: dict[str, tuple[MemoryEntry, ...]],
) -> dict[str, float]:
    label_documents = {
        label: set(label_prototype_tokens(memories))
        for label, memories in grouped_memories.items()
    }
    document_frequency: Counter[str] = Counter()
    for tokens in label_documents.values():
        document_frequency.update(tokens)
    label_count = max(len(label_documents), 1)
    return {
        token: log((label_count + 1) / count) + 1.0
        for token, count in document_frequency.items()
    }


def weighted_overlap_score(
    query_tokens: tuple[str, ...],
    candidate_tokens: tuple[str, ...],
    token_weights: dict[str, float],
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_token_set = set(query_tokens)
    total_weight = 0.0
    matched_weight = 0.0
    for token in candidate_tokens:
        weight = token_weights.get(token, 1.0)
        total_weight += weight
        if token in query_token_set:
            matched_weight += weight
    if total_weight == 0.0:
        return 0.0
    return matched_weight / total_weight


def split_channel_tokens(
    tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    primary = tuple(
        token
        for token in tokens
        if not token.startswith("sensory:") and not token.startswith("feedback:")
    )
    sensory = tuple(token for token in tokens if token.startswith("sensory:"))
    feedback = tuple(token for token in tokens if token.startswith("feedback:"))
    return primary, sensory, feedback


def primary_signal_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    primary, _, _ = split_channel_tokens(tokens)
    return primary


def channel_overlap_score(
    query_tokens: tuple[str, ...],
    candidate_tokens: tuple[str, ...],
    token_weights: dict[str, float],
) -> float:
    query_primary, query_sensory, query_feedback = split_channel_tokens(query_tokens)
    candidate_primary, candidate_sensory, candidate_feedback = split_channel_tokens(candidate_tokens)
    primary_score = weighted_overlap_score(query_primary, candidate_primary, token_weights)
    sensory_score = weighted_overlap_score(query_sensory, candidate_sensory, token_weights)
    feedback_score = weighted_overlap_score(query_feedback, candidate_feedback, token_weights)
    return (
        primary_score
        + (SENSORY_CHANNEL_MATCH_WEIGHT * sensory_score)
        + (FEEDBACK_CHANNEL_MATCH_WEIGHT * feedback_score)
    )


def cosine_similarity(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def memory_numeric_match_score(
    numeric_fingerprint: tuple[int, ...],
    memory: MemoryEntry,
) -> float:
    return cosine_similarity(numeric_fingerprint, memory.numeric_fingerprint)


def label_match_score(
    input_tokens: tuple[str, ...],
    memories: tuple[MemoryEntry, ...],
    token_weights: dict[str, float],
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> float:
    prototype_tokens = label_prototype_tokens(memories)
    prototype_score = channel_overlap_score(
        input_tokens,
        prototype_tokens,
        token_weights,
    )
    exemplar_scores = tuple(
        sorted(
            [
                channel_overlap_score(input_tokens, memory.input_tokens, token_weights)
                + (NUMERIC_FINGERPRINT_MATCH_WEIGHT * memory_numeric_match_score(numeric_fingerprint, memory))
                + (0.01 * memory.weight)
                for memory in memories
            ],
            reverse=True,
        ),
    )
    exemplar_score = (
        sum(exemplar_scores[:2]) / min(len(exemplar_scores), 2)
        if exemplar_scores
        else 0.0
    )
    return exemplar_score + (PROTOTYPE_MATCH_WEIGHT * prototype_score)


def feedback_context_tokens(text: str) -> tuple[str, ...]:
    if extract_target_label(text) is not None:
        return ()
    return trim_memory_tokens(tokens_from_pattern(text))


def residual_channel_tokens(
    *,
    base_tokens: tuple[str, ...],
    channel_tokens: tuple[str, ...],
    prefix: str,
    limit: int,
) -> tuple[str, ...]:
    if not channel_tokens:
        return ()
    base_token_set = set(base_tokens)
    residual = tuple(
        token
        for token in channel_tokens
        if token not in base_token_set
    )[:limit]
    return prefixed_tokens(prefix, residual)


def primary_evidence_tokens(input_data: StateEvolutionInput) -> tuple[str, ...]:
    if not input_data.sensory_input.strip():
        return memory_tokens_from_text(input_data.bottom_up)
    if not input_data.bottom_up.strip() or input_data.bottom_up == input_data.sensory_input:
        return memory_tokens_from_text(input_data.sensory_input)
    return bottom_up_message_tokens(input_data.bottom_up)


def primary_evidence_text(input_data: StateEvolutionInput) -> str:
    if input_data.sensory_input.strip():
        return input_data.sensory_input
    return input_data.bottom_up


def input_numeric_fingerprint(
    input_data: StateEvolutionInput,
) -> tuple[int, ...]:
    return numeric_fingerprint_from_text(primary_evidence_text(input_data))


def sensory_channel_memory_tokens(
    input_data: StateEvolutionInput,
) -> tuple[str, ...]:
    if not input_data.sensory_input.strip():
        return ()
    if (
        input_data.bottom_up.strip()
        and input_data.bottom_up != input_data.sensory_input
        and numeric_fingerprint_from_text(input_data.sensory_input)
    ):
        return ()
    return memory_tokens_from_text(input_data.sensory_input)


def input_evidence_tokens(input_data: StateEvolutionInput) -> tuple[str, ...]:
    primary_tokens = primary_evidence_tokens(input_data)
    if not primary_tokens:
        return ()
    sensory_tokens = sensory_channel_memory_tokens(input_data)
    feedback_tokens = feedback_context_tokens(input_data.top_down)
    sensory_residual_tokens = residual_channel_tokens(
        base_tokens=primary_tokens,
        channel_tokens=sensory_tokens,
        prefix="sensory",
        limit=MAX_SENSORY_RESIDUAL_TOKENS,
    )
    feedback_residual_tokens = residual_channel_tokens(
        base_tokens=primary_tokens,
        channel_tokens=feedback_tokens,
        prefix="feedback",
        limit=MAX_FEEDBACK_RESIDUAL_TOKENS,
    )
    return trim_memory_tokens(
        interleave_token_groups(
            (
                primary_tokens,
                sensory_residual_tokens,
                feedback_residual_tokens,
            ),
        ),
    )


def competing_memory_match_score(
    input_tokens: tuple[str, ...],
    memory: MemoryEntry,
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> float:
    token_score = overlap_score(
        primary_signal_tokens(input_tokens),
        primary_signal_tokens(memory.input_tokens),
    )
    numeric_score = memory_numeric_match_score(numeric_fingerprint, memory)
    return max(token_score, numeric_score)


def weaken_memory_entry(memory: MemoryEntry) -> MemoryEntry | None:
    if memory.weight <= 1:
        return None
    return memory_entry_from_tokens(
        memory.input_tokens,
        memory.output_text,
        memory.weight - 1,
        numeric_fingerprint=memory.numeric_fingerprint,
    )


def weaken_competing_memories(
    memories: tuple[MemoryEntry, ...],
    *,
    predicted_output_text: str | None,
    input_tokens: tuple[str, ...],
    numeric_fingerprint: tuple[int, ...] = (),
) -> tuple[MemoryEntry, ...]:
    if predicted_output_text is None:
        return memories
    adjusted = tuple(
        weakened
        for memory in memories
        for weakened in (
            weaken_memory_entry(memory)
            if (
                memory.output_text == predicted_output_text
                and competing_memory_match_score(
                    input_tokens,
                    memory,
                    numeric_fingerprint=numeric_fingerprint,
                ) >= COMPETING_MEMORY_WEAKEN_THRESHOLD
            )
            else memory,
        )
        if weakened is not None
    )
    return compress_memories(adjusted)


def strongest_output_match(
    state: ParsedNeuronState,
    input_tokens: tuple[str, ...],
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> str | None:
    ranked = ranked_output_matches(
        state,
        input_tokens,
        numeric_fingerprint=numeric_fingerprint,
    )
    if not ranked:
        return None
    best_output_text, best_score = ranked[0]
    second_best_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score >= MEMORY_MATCH_THRESHOLD:
        return best_output_text
    if (
        best_score >= WEAK_MEMORY_MATCH_THRESHOLD
        and best_score - second_best_score >= MEMORY_MATCH_MARGIN
    ):
        return best_output_text
    return None


def ranked_output_matches(
    state: ParsedNeuronState,
    input_tokens: tuple[str, ...],
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> tuple[tuple[str, float], ...]:
    grouped_memories = label_memory_groups(state)
    if not grouped_memories:
        return ()
    token_weights = token_specificity_weights(grouped_memories)
    label_scores = tuple(
        (
            label,
            label_match_score(
                input_tokens,
                memories,
                token_weights,
                numeric_fingerprint=numeric_fingerprint,
            ),
        )
        for label, memories in grouped_memories.items()
    )
    filtered_scores = tuple(score for score in label_scores if score[1] > 0.0)
    if not filtered_scores:
        return ()
    return tuple(sorted(filtered_scores, key=lambda item: item[1], reverse=True))


def best_output_match(
    state: ParsedNeuronState,
    input_tokens: tuple[str, ...],
    *,
    numeric_fingerprint: tuple[int, ...] = (),
) -> str | None:
    ranked = ranked_output_matches(
        state,
        input_tokens,
        numeric_fingerprint=numeric_fingerprint,
    )
    if not ranked:
        return None
    return ranked[0][0]


def memory_bank_token_weights(memories: tuple[MemoryEntry, ...]) -> dict[str, float]:
    documents = tuple(set(memory.input_tokens) for memory in memories if memory.input_tokens)
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(tokens)
    document_count = max(len(documents), 1)
    return {
        token: log((document_count + 1) / count) + 1.0
        for token, count in document_frequency.items()
    }


def ranked_memory_matches(
    state: ParsedNeuronState,
    input_tokens: tuple[str, ...],
) -> tuple[tuple[MemoryEntry, float], ...]:
    if not input_tokens:
        return ()
    ranked = tuple(
        (
            memory,
            overlap_score(input_tokens, memory.input_tokens) + min(memory.weight, 8) * 0.01,
        )
        for memory in state.memories
        if memory.input_tokens
    )
    filtered = tuple(item for item in ranked if item[1] > 0.0)
    if not filtered:
        return ()
    return tuple(sorted(filtered, key=lambda item: item[1], reverse=True))


def projected_tokens(state: ParsedNeuronState, input_tokens: tuple[str, ...]) -> tuple[str, ...]:
    input_primary_tokens = trim_protocol_tokens(primary_signal_tokens(input_tokens))
    if not input_primary_tokens:
        return ()
    ranked_matches = ranked_memory_matches(state, input_tokens)
    if not ranked_matches:
        return input_primary_tokens
    token_weights = memory_bank_token_weights(state.memories)
    token_scores: Counter[str] = Counter()
    token_order = list(input_primary_tokens)
    for memory, score in ranked_matches[:MAX_RETRIEVED_MEMORIES]:
        match_scale = score * (1.0 + min(memory.weight, 8) * 0.05)
        memory_token_set = set(primary_signal_tokens(memory.input_tokens))
        for token in input_primary_tokens:
            if token not in memory_token_set:
                continue
            token_scores[token] += match_scale * token_weights.get(token, 1.0)
            token_scores[token] += match_scale
    order_index = {token: index for index, token in enumerate(token_order)}
    ranked_tokens = tuple(
        sorted(
            token_order,
            key=lambda token: (-token_scores[token], order_index[token]),
        ),
    )
    return trim_protocol_tokens(ranked_tokens)


def target_feedback_tokens(
    state: ParsedNeuronState,
    output_text: str,
    input_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    grouped_memories = label_memory_groups(state)
    matching = grouped_memories.get(output_text, ())
    if not matching:
        return trim_protocol_tokens(primary_signal_tokens(input_tokens))
    token_weights = token_specificity_weights(grouped_memories)
    prototype_tokens = primary_signal_tokens(label_prototype_tokens(matching))
    order_index = {token: index for index, token in enumerate(prototype_tokens)}
    ranked_tokens = tuple(
        sorted(
            prototype_tokens,
            key=lambda token: (-token_weights.get(token, 1.0), order_index[token]),
        ),
    )
    return trim_protocol_tokens(ranked_tokens)


def contrastive_feedback_tokens(
    state: ParsedNeuronState,
    *,
    output_text: str,
    predicted_output_text: str | None,
    input_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    target_tokens = target_feedback_tokens(state, output_text, input_tokens)
    if predicted_output_text is None or predicted_output_text == output_text:
        return target_tokens
    competing_tokens = set(
        target_feedback_tokens(state, predicted_output_text, input_tokens)
    )
    contrastive_tokens = tuple(
        token
        for token in target_tokens
        if token not in competing_tokens
    )
    if contrastive_tokens:
        return trim_protocol_tokens(contrastive_tokens)
    return target_tokens


def contrastive_training_tokens(
    state: ParsedNeuronState,
    *,
    output_text: str,
    predicted_output_text: str | None,
    input_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    input_primary_tokens = trim_memory_tokens(primary_signal_tokens(input_tokens))
    if not input_primary_tokens:
        return ()
    if predicted_output_text is None or predicted_output_text == output_text:
        return input_primary_tokens
    target_tokens = contrastive_feedback_tokens(
        state,
        output_text=output_text,
        predicted_output_text=predicted_output_text,
        input_tokens=input_tokens,
    )
    predicted_tokens = set(
        target_feedback_tokens(state, predicted_output_text, input_tokens)
    )
    residual_tokens = tuple(
        token
        for token in input_primary_tokens
        if token not in predicted_tokens
    )
    supported_target_tokens = tuple(
        token
        for token in target_tokens
        if token in set(input_primary_tokens)
    )
    contrastive_tokens = interleave_token_groups(
        (
            supported_target_tokens,
            target_tokens,
            residual_tokens,
        ),
    )
    if contrastive_tokens:
        return trim_memory_tokens(contrastive_tokens)
    return input_primary_tokens


def best_support_memory_match(
    state: ParsedNeuronState,
    numeric_fingerprint: tuple[int, ...],
) -> tuple[str, float] | None:
    if not numeric_fingerprint:
        return None
    best_label = None
    best_score = 0.0
    for support_memory in state.support_memories:
        score = cosine_similarity(
            numeric_fingerprint,
            support_memory.numeric_fingerprint,
        )
        if score <= best_score:
            continue
        best_label = support_memory.output_text
        best_score = score
    if best_label is None:
        return None
    return best_label, best_score


def evolve_state(input_data: StateEvolutionInput) -> ParsedNeuronState:
    state = parse_state_text(input_data.previous_state_text)
    memories = state.memories
    support_memories = state.support_memories
    input_tokens = input_evidence_tokens(input_data)
    numeric_fingerprint = input_numeric_fingerprint(input_data)
    memories = add_memory_tokens(
        memories,
        input_tokens,
        numeric_fingerprint=numeric_fingerprint,
    )
    if extract_target_label(input_data.top_down) is None:
        memories = add_memory(memories, input_data.top_down)
    for text in input_data.candidate_texts:
        memories = add_memory(memories, text)

    target_output_text = extract_target_label(input_data.top_down)
    if target_output_text is not None:
        support_memories = add_support_memory(
            support_memories,
            numeric_fingerprint=numeric_fingerprint,
            output_text=target_output_text,
        )
        memories = add_memory_tokens(
            memories,
            input_tokens,
            target_output_text,
            numeric_fingerprint=numeric_fingerprint,
        )
        predicted_output_text = extract_predicted_label(input_data.top_down)
        if predicted_output_text is None or predicted_output_text == target_output_text:
            predicted_output_text = None
        contrastive_tokens = contrastive_training_tokens(
            state,
            output_text=target_output_text,
            predicted_output_text=predicted_output_text,
            input_tokens=input_tokens,
        )
        if predicted_output_text is not None:
            memories = add_memory_tokens(
                memories,
                contrastive_tokens,
                target_output_text,
                numeric_fingerprint=numeric_fingerprint,
            )
        memories = weaken_competing_memories(
            memories,
            predicted_output_text=predicted_output_text,
            input_tokens=input_tokens,
            numeric_fingerprint=numeric_fingerprint,
        )

    next_state = ParsedNeuronState(
        summary_tokens=state.summary_tokens,
        memories=memories,
        support_memories=support_memories,
    )
    return ParsedNeuronState(
        summary_tokens=recompute_summary(next_state),
        memories=memories,
        support_memories=support_memories,
    )


def choose_activation(input_data: StateEvolutionInput, state: ParsedNeuronState) -> str:
    target_output_text = extract_target_label(input_data.top_down)
    if target_output_text is not None:
        return target_output_text

    numeric_fingerprint = input_numeric_fingerprint(input_data)
    support_match = best_support_memory_match(state, numeric_fingerprint)
    if support_match is not None:
        return support_match[0]

    preferred_output = strongest_output_match(
        state,
        tokens_from_pattern(input_data.preferred_activation),
        numeric_fingerprint=numeric_fingerprint,
    )
    if preferred_output is not None:
        return preferred_output

    input_tokens = input_evidence_tokens(input_data)
    matched_output = strongest_output_match(
        state,
        input_tokens,
        numeric_fingerprint=numeric_fingerprint,
    )
    if matched_output is not None:
        return matched_output
    fallback_output = best_output_match(
        state,
        input_tokens,
        numeric_fingerprint=numeric_fingerprint,
    )
    if fallback_output is not None:
        return fallback_output

    projected = projected_tokens(state, input_tokens)
    if projected:
        return build_pattern(projected)

    preferred_tokens = trim_protocol_tokens(tokens_from_pattern(input_data.preferred_activation))
    if preferred_tokens:
        return build_pattern(preferred_tokens)
    return "unknown"


def choose_feedback(input_data: StateEvolutionInput, state: ParsedNeuronState) -> str:
    target_output_text = extract_target_label(input_data.top_down)
    input_tokens = input_evidence_tokens(input_data)
    if target_output_text is not None:
        return build_pattern(
            contrastive_feedback_tokens(
                state,
                output_text=target_output_text,
                predicted_output_text=extract_predicted_label(input_data.top_down),
                input_tokens=input_tokens,
            ),
        )

    preferred_tokens = trim_protocol_tokens(tokens_from_pattern(input_data.preferred_feedback))
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
    sensory_input: str = "",
    preferred_activation: str = "",
    preferred_feedback: str = "",
) -> tuple[str, str]:
    input_data = StateEvolutionInput(
        previous_state_text=serialize_state(state),
        bottom_up=bottom_up,
        top_down=top_down,
        sensory_input=sensory_input,
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
