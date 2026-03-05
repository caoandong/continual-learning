#!/usr/bin/env python3
"""Test the SPCA network on the MNIST continual learning problem.

Trains the network in incremental phases, adding new digits over time,
then evaluates whether previously learned digits are retained.

Usage:
    uv run scripts/test_mnist.py                    # quick mock run
    uv run scripts/test_mnist.py --rounds 5         # 5 training rounds per digit
    uv run scripts/test_mnist.py --digits 0,1,2,3   # learn 4 digits
    uv run scripts/test_mnist.py --model gpt-5.2    # use a real LLM
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from continual_learning.constants import DEFAULT_LAYER_SIZES, DEFAULT_MODEL
from continual_learning.experiment import evaluate_sample, train_on_sample
from continual_learning.llm import create_batch_llm_caller
from continual_learning.network import create_network_state
from continual_learning.types import (
    BatchLlmCaller,
    ExperimentOptions,
    LayerState,
    NetworkState,
)

logger = logging.getLogger(__name__)

DIGIT_FEATURES: dict[int, str] = {
    0: "round closed loop",
    1: "straight vertical line",
    2: "top curve diagonal flat bottom",
}

DIGIT_VARIANTS: dict[int, tuple[str, ...]] = {
    0: (
        DIGIT_FEATURES[0],
        "closed round loop",
        "loop closed round",
        "round loop closed",
    ),
    1: (
        DIGIT_FEATURES[1],
        "vertical straight line",
        "line straight vertical",
        "straight line vertical",
    ),
    2: (
        DIGIT_FEATURES[2],
        "flat bottom diagonal curve top",
        "top flat curve bottom diagonal",
        "curve top diagonal flat bottom",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configure_logging(*, verbose: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    for noisy in ("httpx", "LiteLLM", "litellm", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def evaluate_digits(
    state: NetworkState, digits: Sequence[int], call_llm_batch: BatchLlmCaller,
) -> tuple[NetworkState, int, int]:
    correct = 0
    total = 0
    current = state
    for digit in digits:
        for features in DIGIT_VARIANTS.get(digit, (DIGIT_FEATURES[digit],)):
            current, prediction = evaluate_sample(current, features, call_llm_batch)
            is_correct = prediction == f"{digit}"
            if is_correct:
                correct += 1
            total += 1
            status = "correct" if is_correct else "WRONG"
            logger.info(
                "  digit %d | input: %-36s | pred: %-12s | %s",
                digit, features, prediction, status,
            )
    return current, correct, total


def inspect_neuron_states(state: NetworkState) -> None:
    for layer_index, layer in enumerate(state.layers):
        for neuron in layer.neurons:
            logger.info("  %s: %s", neuron.name, neuron.state or "(empty)")


def remove_neuron(state: NetworkState, layer_index: int, neuron_index: int) -> NetworkState:
    layer = state.layers[layer_index]
    remaining = layer.neurons[:neuron_index] + layer.neurons[neuron_index + 1:]
    new_layer = LayerState(neurons=remaining)

    new_layers = state.layers[:layer_index] + (new_layer,) + state.layers[layer_index + 1:]
    new_activations = (
        state.activations[:layer_index]
        + (state.activations[layer_index][:neuron_index]
           + state.activations[layer_index][neuron_index + 1:],)
        + state.activations[layer_index + 1:]
    )
    new_feedbacks = (
        state.feedbacks[:layer_index]
        + (state.feedbacks[layer_index][:neuron_index]
           + state.feedbacks[layer_index][neuron_index + 1:],)
        + state.feedbacks[layer_index + 1:]
    )
    return NetworkState(layers=new_layers, activations=new_activations, feedbacks=new_feedbacks)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(
    digits: list[int],
    rounds: int,
    layer_sizes: tuple[int, ...],
    call_llm_batch: BatchLlmCaller,
) -> None:
    state = create_network_state(layer_sizes)
    logger.info("Network: layers=%s  digits=%s  rounds=%d", layer_sizes, digits, rounds)
    logger.info("=== Initial evaluation on all requested digit variants ===")
    _, correct, total = evaluate_digits(state, digits, call_llm_batch)
    logger.info("  initial accuracy: %d/%d", correct, total)

    learned_so_far: list[int] = []

    # Incremental training: add one digit at a time
    for digit in digits:
        learned_so_far.append(digit)
        logger.info("--- Training on digit %d (%d rounds) ---", digit, rounds)

        features = DIGIT_FEATURES[digit]
        label = f"{digit}"
        for round_number in range(1, rounds + 1):
            state = train_on_sample(state, features, label, call_llm_batch)
            _, prediction = evaluate_sample(state, features, call_llm_batch)
            status = "correct" if prediction == label else "wrong"
            logger.info(
                "  round %d/%d | pred: %-12s | %s",
                round_number, rounds, prediction, status,
            )

        # Evaluate all learned digits after each new digit is introduced
        logger.info("--- Evaluation on digits %s ---", learned_so_far)
        state, correct, total = evaluate_digits(state, learned_so_far, call_llm_batch)
        logger.info("  accuracy: %d/%d", correct, total)

    # Final full evaluation
    logger.info("=== Final evaluation on all digits %s ===", digits)
    state, correct, total = evaluate_digits(state, digits, call_llm_batch)
    logger.info("  final accuracy: %d/%d", correct, total)

    # Neuron state inspection
    logger.info("=== Learned neuron states ===")
    inspect_neuron_states(state)

    # Topology test: remove first neuron from layer 0
    if len(state.layers[0].neurons) >= 2:
        logger.info("=== Topology test: removing L0_N0 ===")
        damaged = remove_neuron(state, layer_index=0, neuron_index=0)
        _, prediction = evaluate_sample(damaged, DIGIT_VARIANTS[digits[0]][-1], call_llm_batch)
        logger.info(
            "  post-damage prediction for digit %d variant: %s (no crash)",
            digits[0], prediction,
        )


def parse_digits(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",")]
    for digit in result:
        if digit not in DIGIT_FEATURES:
            raise argparse.ArgumentTypeError(
                "digit %d not supported (available: %s)"
                % (digit, sorted(DIGIT_FEATURES)),
            )
    return result


def parse_layer_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SPCA on MNIST continual learning")
    parser.add_argument(
        "--digits", type=parse_digits, default=[0, 1, 2],
        help="Comma-separated digits to learn incrementally (default: 0,1,2)",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Training rounds per digit (default: 1)",
    )
    parser.add_argument(
        "--layer-sizes", type=parse_layer_sizes, default=DEFAULT_LAYER_SIZES,
        help="Comma-separated layer sizes (default: 2,1)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="LLM model name (default: %(default)s)",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    # Default to mock when no model override is given explicitly
    use_mock = args.mock or "--model" not in sys.argv
    options = ExperimentOptions(
        model=args.model, use_mock=use_mock,
        layer_sizes=args.layer_sizes, verbose=args.verbose,
    )
    call_llm_batch = create_batch_llm_caller(options)
    run_test(args.digits, args.rounds, args.layer_sizes, call_llm_batch)


if __name__ == "__main__":
    main()
