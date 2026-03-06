from __future__ import annotations

from continual_learning.experiment import evaluate_sample, train_on_sample
from continual_learning.llm import call_llm_mock
from continual_learning.network import create_network_state
from continual_learning.types import NeuronResponse


def call_llm_mock_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
    return tuple(call_llm_mock(prompt) for prompt in prompts)


def test_evaluate_sample_does_not_mutate_persistent_state() -> None:
    state = create_network_state((1, 1))
    trained_state = train_on_sample(
        state,
        "alpha beta gamma",
        "class_a",
        call_llm_mock_batch,
    )
    state_after_eval, prediction = evaluate_sample(
        trained_state,
        "gamma alpha beta",
        call_llm_mock_batch,
    )
    assert prediction == "class_a"
    assert state_after_eval == trained_state


def test_train_and_evaluate_supports_deeper_networks() -> None:
    state = create_network_state((2, 2, 1))
    trained_state = train_on_sample(
        state,
        "alpha beta gamma",
        "class_a",
        call_llm_mock_batch,
    )
    _, prediction = evaluate_sample(
        trained_state,
        "gamma beta alpha",
        call_llm_mock_batch,
    )
    assert prediction == "class_a"


def test_train_and_evaluate_supports_four_layer_networks() -> None:
    state = create_network_state((3, 2, 2, 1))
    trained_state = train_on_sample(
        state,
        "alpha beta gamma",
        "class_a",
        call_llm_mock_batch,
    )
    _, prediction = evaluate_sample(
        trained_state,
        "gamma beta alpha",
        call_llm_mock_batch,
    )
    assert prediction == "class_a"
