from __future__ import annotations

import logging
from collections.abc import Iterator

from continual_learning.constants import (
    DEFAULT_LAYER_SIZES,
    ERROR_CORRECTION_TICKS,
    PROPAGATION_TICKS,
)
from continual_learning.network import create_network_state, step_network
from continual_learning.types import (
    ExperimentOptions,
    LlmCaller,
    LayerState,
    NetworkState,
    NetworkStepInput,
)

logger = logging.getLogger(__name__)

DIGIT_FEATURES: dict[int, str] = {
    0: "Round closed loop",
    1: "Straight vertical line",
    2: "Top curve, diagonal, flat bottom",
}


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_mnist_samples(digits: list[int]) -> Iterator[tuple[str, int]]:
    for digit in digits:
        yield DIGIT_FEATURES[digit], digit


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_on_sample(
    state: NetworkState, features: str, target: int, call_llm: LlmCaller,
) -> NetworkState:
    current = state
    for _ in range(PROPAGATION_TICKS):
        result = step_network(
            current,
            NetworkStepInput(raw_input=features, top_down_feedback="Evaluate"),
            call_llm,
        )
        current = result.state

    prediction = result.prediction  # noqa: F821 — `result` is always assigned
    if prediction != str(target):
        feedback = f"Error: Target: {target}"
        for _ in range(ERROR_CORRECTION_TICKS):
            result = step_network(
                current,
                NetworkStepInput(raw_input=features, top_down_feedback=feedback),
                call_llm,
            )
            current = result.state
    return current


def evaluate_sample(
    state: NetworkState, features: str, call_llm: LlmCaller,
) -> tuple[NetworkState, str]:
    current = state
    prediction = "unknown"
    for _ in range(PROPAGATION_TICKS):
        result = step_network(
            current,
            NetworkStepInput(raw_input=features, top_down_feedback="Evaluate"),
            call_llm,
        )
        current = result.state
        prediction = result.prediction
    return current, prediction


# ---------------------------------------------------------------------------
# Experiment phases
# ---------------------------------------------------------------------------

def run_training_phase(
    state: NetworkState,
    digits: list[int],
    call_llm: LlmCaller,
    phase_name: str,
) -> NetworkState:
    logger.info("[experiment] === %s ===", phase_name)
    current = state
    for features, target in generate_mnist_samples(digits):
        current = train_on_sample(current, features, target, call_llm)
        _, prediction = evaluate_sample(current, features, call_llm)
        status = "correct" if prediction == str(target) else "wrong"
        logger.info(
            "[experiment] Train | Input: '%s' | Target: %d | Pred: %s | %s",
            features, target, prediction, status,
        )
    return current


def run_evaluation_phase(
    state: NetworkState,
    digits: list[int],
    call_llm: LlmCaller,
    phase_name: str,
) -> NetworkState:
    logger.info("[experiment] === %s ===", phase_name)
    current = state
    correct = 0
    total = 0
    for features, target in generate_mnist_samples(digits):
        current, prediction = evaluate_sample(current, features, call_llm)
        is_correct = prediction == str(target)
        if is_correct:
            correct += 1
        total += 1
        status = "correct" if is_correct else "FORGOTTEN"
        logger.info(
            "[experiment] Eval  | Input: '%s' | Target: %d | Pred: %s | %s",
            features, target, prediction, status,
        )
    logger.info("[experiment] Accuracy: %d/%d", correct, total)
    return current


def run_topology_phase(
    state: NetworkState,
    call_llm: LlmCaller,
) -> None:
    logger.info("[experiment] === Phase 4: Dynamic Topology (Neuron Removal) ===")
    layer_0 = state.layers[0]
    if len(layer_0.neurons) < 2:
        logger.info("[experiment] Layer 0 has fewer than 2 neurons, skipping removal")
        return

    logger.info("[experiment] Removing sensory neuron L0_N0 (simulating cell death)")
    reduced_layer = LayerState(neurons=layer_0.neurons[1:])
    reduced_layers = (reduced_layer,) + state.layers[1:]
    reduced_activations = (state.activations[0][1:],) + state.activations[1:]
    reduced_feedbacks = (state.feedbacks[0][1:],) + state.feedbacks[1:]
    reduced_state = NetworkState(
        layers=reduced_layers,
        activations=reduced_activations,
        feedbacks=reduced_feedbacks,
    )

    result = step_network(
        reduced_state,
        NetworkStepInput(raw_input="Straight vertical line", top_down_feedback="Evaluate"),
        call_llm,
    )
    logger.info(
        "[experiment] Post-damage prediction: %s (no shape crash!)", result.prediction,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(options: ExperimentOptions, call_llm: LlmCaller) -> None:
    layer_sizes = options.layer_sizes or DEFAULT_LAYER_SIZES
    state = create_network_state(layer_sizes)
    logger.info("[experiment] Network created with layers: %s", layer_sizes)

    # Phase 1: Learn digits 0 and 1
    state = run_training_phase(state, [0, 1], call_llm, "Phase 1: Learning 0 and 1")

    # Phase 2: Introduce digit 2 (continual learning)
    state = run_training_phase(state, [2], call_llm, "Phase 2: Introducing digit 2")

    # Phase 3: Test all digits (verify no catastrophic forgetting)
    state = run_evaluation_phase(
        state, [0, 1, 2], call_llm, "Phase 3: Testing all digits (forgetting check)",
    )

    # Phase 4: Dynamic topology — neuron removal
    run_topology_phase(state, call_llm)

    # Print final learned state
    logger.info("[experiment] Final learned tokens of output neuron:")
    output_neuron = state.layers[-1].neurons[0]
    logger.info("[experiment] %s", output_neuron.state)
