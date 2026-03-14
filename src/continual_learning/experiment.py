from __future__ import annotations

import logging

from continual_learning.constants import (
    DEFAULT_READ_ONLY_SETTLE_PASS_COUNT,
    DEFAULT_TRAIN_CONSISTENCY_ATTEMPTS,
    EMPTY_SIGNAL,
)
from continual_learning.network import reset_network_traces, step_network
from continual_learning.state import chunk_input_text, render_signal
from continual_learning.types import (
    BatchLlmCaller,
    NetworkReadout,
    NetworkState,
    NetworkStepInput,
)

logger = logging.getLogger(__name__)


def propagation_flush_count(layer_count: int) -> int:
    return max(0, layer_count - 1)


def build_pass_inputs(
    raw_input: str,
    *,
    layer_count: int,
    allow_state_update: bool,
    teaching_signal: str = EMPTY_SIGNAL,
) -> tuple[NetworkStepInput, ...]:
    chunk_steps = tuple(
        NetworkStepInput(
            raw_input=chunk,
            teaching_signal=teaching_signal,
            allow_state_update=allow_state_update,
        )
        for chunk in chunk_input_text(raw_input)
    )
    flush_steps = tuple(
        NetworkStepInput(
            raw_input=EMPTY_SIGNAL,
            teaching_signal=teaching_signal,
            allow_state_update=allow_state_update,
        )
        for _ in range(propagation_flush_count(layer_count))
    )
    return chunk_steps + flush_steps


def run_pass(
    state: NetworkState,
    *,
    raw_input: str,
    allow_state_update: bool,
    call_llm_batch: BatchLlmCaller,
    teaching_signal: str = EMPTY_SIGNAL,
    log_prefix: str,
) -> tuple[NetworkState, NetworkReadout]:
    current = state
    readout = NetworkReadout()
    for index, step_input in enumerate(
        build_pass_inputs(
            raw_input,
            layer_count=len(state.layers),
            allow_state_update=allow_state_update,
            teaching_signal=teaching_signal,
        ),
        start=1,
    ):
        result = step_network(current, step_input, call_llm_batch)
        current = result.state
        readout = NetworkReadout(
            latent_output=result.latent_output,
            task_output=result.task_output,
        )
        logger.debug(
            "[experiment] %s step=%d latent=%s task=%s",
            log_prefix,
            index,
            render_signal(readout.latent_output),
            render_signal(readout.task_output),
        )
    return current, readout


def settle_read_only(
    state: NetworkState,
    *,
    raw_input: str,
    call_llm_batch: BatchLlmCaller,
    log_prefix: str,
) -> tuple[NetworkState, NetworkReadout]:
    current = state
    readout = NetworkReadout()
    previous_task_output = None
    for pass_index in range(DEFAULT_READ_ONLY_SETTLE_PASS_COUNT):
        current, readout = run_pass(
            current,
            raw_input=raw_input,
            allow_state_update=False,
            call_llm_batch=call_llm_batch,
            log_prefix=f"{log_prefix}[{pass_index + 1}]",
        )
        if pass_index > 0 and readout.task_output == previous_task_output:
            logger.debug(
                "[experiment] %s settled after pass=%d latent=%s task=%s",
                log_prefix,
                pass_index + 1,
                render_signal(readout.latent_output),
                render_signal(readout.task_output),
            )
            break
        previous_task_output = readout.task_output
    return current, readout


def train_on_sample(
    state: NetworkState,
    features: str,
    target_label: str,
    call_llm_batch: BatchLlmCaller,
) -> NetworkState:
    logger.debug(
        "[experiment] train_on_sample START features=%s target=%s",
        render_signal(features),
        render_signal(target_label),
    )
    current = state
    inspect_readout = NetworkReadout()
    for attempt in range(DEFAULT_TRAIN_CONSISTENCY_ATTEMPTS):
        primed, _ = settle_read_only(
            reset_network_traces(current),
            raw_input=features,
            call_llm_batch=call_llm_batch,
            log_prefix=f"train_on_sample READ_ONLY[{attempt + 1}]",
        )
        current, _ = run_pass(
            primed,
            raw_input=features,
            allow_state_update=True,
            teaching_signal=target_label,
            call_llm_batch=call_llm_batch,
            log_prefix=f"train_on_sample WRITE_SUPERVISED[{attempt + 1}]",
        )
        _, inspect_readout = settle_read_only(
            reset_network_traces(current),
            raw_input=features,
            call_llm_batch=call_llm_batch,
            log_prefix=f"train_on_sample INSPECT[{attempt + 1}]",
        )
        if inspect_readout.task_output == target_label:
            break
        logger.debug(
            "[experiment] train_on_sample retry attempt=%d latent=%s task=%s target=%s",
            attempt + 1,
            render_signal(inspect_readout.latent_output),
            render_signal(inspect_readout.task_output),
            render_signal(target_label),
        )
    logger.debug(
        "[experiment] train_on_sample END latent=%s task=%s",
        render_signal(inspect_readout.latent_output),
        render_signal(inspect_readout.task_output),
    )
    return current


def evaluate_sample(
    state: NetworkState,
    features: str,
    call_llm_batch: BatchLlmCaller,
) -> tuple[NetworkState, NetworkReadout]:
    logger.debug("[experiment] evaluate_sample START features=%s", render_signal(features))
    _, readout = settle_read_only(
        reset_network_traces(state),
        raw_input=features,
        call_llm_batch=call_llm_batch,
        log_prefix="evaluate_sample READ_ONLY",
    )
    logger.debug(
        "[experiment] evaluate_sample END latent=%s task=%s",
        render_signal(readout.latent_output),
        render_signal(readout.task_output),
    )
    return state, readout
