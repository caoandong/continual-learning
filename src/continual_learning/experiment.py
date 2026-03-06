from __future__ import annotations

import logging
from dataclasses import dataclass

from continual_learning.constants import (
    ERROR_CORRECTION_TICKS,
    MAX_CORRECTION_PASSES,
    PROPAGATION_TICKS,
    SETTLING_BUFFER_TICKS,
)
from continual_learning.network import (
    format_network_state,
    reset_network_traces,
    step_network,
)
from continual_learning.types import (
    BatchLlmCaller,
    NetworkState,
    NetworkStepInput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseResult:
    state: NetworkState
    prediction: str
    ticks: int


def propagation_tick_count(state: NetworkState) -> int:
    return max(PROPAGATION_TICKS, len(state.layers))


def correction_tick_count(state: NetworkState) -> int:
    return max(ERROR_CORRECTION_TICKS, len(state.layers))


def traces_are_stable(previous: NetworkState, current: NetworkState) -> bool:
    return (
        previous.activations == current.activations
        and previous.feedbacks == current.feedbacks
    )


def settle_phase(
    state: NetworkState,
    *,
    features: str,
    feedback: str,
    call_llm_batch: BatchLlmCaller,
    tick_budget: int,
    log_prefix: str,
) -> PhaseResult:
    current = state
    minimum_ticks = len(state.layers)
    for tick in range(tick_budget + SETTLING_BUFFER_TICKS):
        result = step_network(
            current,
            NetworkStepInput(
                raw_input=features,
                top_down_feedback=feedback,
                allow_state_update=False,
            ),
            call_llm_batch,
        )
        next_state = result.state
        stabilized = (
            tick + 1 >= minimum_ticks
            and traces_are_stable(current, next_state)
        )
        logger.debug(
            "[experiment] %s SETTLE tick %d/%d features=%s feedback=%s prediction=%s stabilized=%s",
            log_prefix,
            tick + 1,
            tick_budget + SETTLING_BUFFER_TICKS,
            features,
            feedback,
            result.prediction,
            stabilized,
        )
        current = next_state
        if stabilized:
            return PhaseResult(
                state=current,
                prediction=result.prediction,
                ticks=tick + 1,
            )
    return PhaseResult(
        state=current,
        prediction=result.prediction,
        ticks=tick_budget + SETTLING_BUFFER_TICKS,
    )


def commit_phase(
    state: NetworkState,
    *,
    features: str,
    feedback: str,
    call_llm_batch: BatchLlmCaller,
    log_prefix: str,
) -> PhaseResult:
    result = step_network(
        state,
        NetworkStepInput(
            raw_input=features,
            top_down_feedback=feedback,
            allow_state_update=True,
        ),
        call_llm_batch,
    )
    logger.debug(
        "[experiment] %s COMMIT features=%s feedback=%s prediction=%s",
        log_prefix,
        features,
        feedback,
        result.prediction,
    )
    return PhaseResult(
        state=result.state,
        prediction=result.prediction,
        ticks=1,
    )


def predict_from_state(
    state: NetworkState,
    *,
    features: str,
    call_llm_batch: BatchLlmCaller,
    log_prefix: str,
) -> str:
    settled = settle_phase(
        reset_network_traces(state),
        features=features,
        feedback="Evaluate",
        call_llm_batch=call_llm_batch,
        tick_budget=propagation_tick_count(state),
        log_prefix=log_prefix,
    )
    return settled.prediction


def correction_feedback(*, target_label: str, prediction: str) -> str:
    if prediction and prediction != "unknown":
        return f"Error: Predicted: {prediction} | Target: {target_label}"
    return f"Error: Target: {target_label}"


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_on_sample(
    state: NetworkState, features: str, target_label: str, call_llm_batch: BatchLlmCaller,
) -> NetworkState:
    logger.debug(
        "[experiment] train_on_sample START features=%s target=%s",
        features, target_label,
    )
    current = reset_network_traces(state)
    propagation = settle_phase(
        current,
        features=features,
        feedback="Evaluate",
        call_llm_batch=call_llm_batch,
        tick_budget=propagation_tick_count(current),
        log_prefix="train_on_sample PROPAGATION",
    )
    current = commit_phase(
        propagation.state,
        features=features,
        feedback="Evaluate",
        call_llm_batch=call_llm_batch,
        log_prefix="train_on_sample PROPAGATION",
    ).state
    prediction = predict_from_state(
        current,
        features=features,
        call_llm_batch=call_llm_batch,
        log_prefix="train_on_sample VERIFY",
    )
    logger.debug(
        "[experiment] train_on_sample prediction=%s target=%s match=%s",
        prediction, target_label, prediction == target_label,
    )
    supervision_pass = 0
    while supervision_pass < MAX_CORRECTION_PASSES:
        if supervision_pass > 0 and prediction == target_label:
            break
        supervision_pass += 1
        feedback = correction_feedback(
            target_label=target_label,
            prediction=prediction,
        )
        logger.debug(
            "[experiment] train_on_sample SUPERVISED UPDATE pass=%d/%d feedback=%s",
            supervision_pass,
            MAX_CORRECTION_PASSES,
            feedback,
        )
        correction = settle_phase(
            current,
            features=features,
            feedback=feedback,
            call_llm_batch=call_llm_batch,
            tick_budget=correction_tick_count(current),
            log_prefix="train_on_sample ERROR_CORRECTION",
        )
        current = commit_phase(
            correction.state,
            features=features,
            feedback=feedback,
            call_llm_batch=call_llm_batch,
            log_prefix="train_on_sample ERROR_CORRECTION",
        ).state
        prediction = predict_from_state(
            current,
            features=features,
            call_llm_batch=call_llm_batch,
            log_prefix="train_on_sample VERIFY",
        )
        logger.debug(
            "[experiment] train_on_sample SUPERVISED UPDATE pass=%d prediction=%s",
            supervision_pass,
            prediction,
        )
    logger.debug(
        "[experiment] train_on_sample END features=%s target=%s\n"
        "  NETWORK STATE:\n%s",
        features, target_label, format_network_state(current),
    )
    return current


def evaluate_sample(
    state: NetworkState, features: str, call_llm_batch: BatchLlmCaller,
) -> tuple[NetworkState, str]:
    logger.debug("[experiment] evaluate_sample START features=%s", features)
    working_state = reset_network_traces(state)
    propagation = settle_phase(
        working_state,
        features=features,
        feedback="Evaluate",
        call_llm_batch=call_llm_batch,
        tick_budget=propagation_tick_count(working_state),
        log_prefix="evaluate_sample PROPAGATION",
    )
    logger.debug(
        "[experiment] evaluate_sample END features=%s prediction=%s",
        features, propagation.prediction,
    )
    return state, propagation.prediction
