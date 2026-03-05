from __future__ import annotations

from continual_learning.constants import EMPTY_SIGNAL
from continual_learning.llm import call_llm_mock
from continual_learning.network import (
    apply_all_results,
    build_layer_bottom_up,
    build_layer_top_down,
    collect_all_requests,
    create_network_state,
    reset_network_traces,
    step_network,
)
from continual_learning.state import parse_state_text
from continual_learning.types import (
    LayerState,
    NetworkState,
    NetworkStepInput,
    NeuronCallResult,
    NeuronResponse,
)


def call_llm_mock_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
    return tuple(call_llm_mock(prompt) for prompt in prompts)


def test_create_network_state_default() -> None:
    state = create_network_state((2, 1))
    assert len(state.layers) == 2
    assert len(state.layers[0].neurons) == 2
    assert len(state.layers[1].neurons) == 1
    assert state.layers[0].neurons[0].name == "L0_N0"
    assert parse_state_text(state.layers[0].neurons[0].state).summary_tokens


def test_create_network_state_buffers_initialized() -> None:
    state = create_network_state((3, 2))
    assert len(state.activations[0]) == 3
    assert len(state.activations[1]) == 2
    assert all(activation == EMPTY_SIGNAL for activation in state.activations[0])
    assert all(feedback == EMPTY_SIGNAL for feedback in state.feedbacks[1])


def test_create_network_state_uses_distinct_randomized_summaries() -> None:
    state = create_network_state((2, 1))
    summaries = {
        parse_state_text(neuron.state).summary_tokens
        for layer in state.layers
        for neuron in layer.neurons
    }
    assert len(summaries) > 1


def test_build_layer_bottom_up_layer_zero() -> None:
    state = create_network_state((2, 1))
    assert build_layer_bottom_up(state, 0, "raw pixels") == "raw pixels"


def test_build_layer_bottom_up_higher_layer() -> None:
    state = create_network_state((2, 1))
    assert build_layer_bottom_up(state, 1, "raw pixels") == f"{EMPTY_SIGNAL} | {EMPTY_SIGNAL}"


def test_build_layer_top_down_top_layer() -> None:
    state = create_network_state((2, 1))
    assert build_layer_top_down(state, 1, "Evaluate") == "Evaluate"


def test_build_layer_top_down_lower_layer() -> None:
    state = create_network_state((2, 1))
    assert build_layer_top_down(state, 0, "Evaluate") == EMPTY_SIGNAL


def test_step_network_returns_prediction() -> None:
    state = create_network_state((2, 1))
    result = step_network(
        state,
        NetworkStepInput(raw_input="alpha beta gamma", top_down_feedback="Evaluate"),
        call_llm_mock_batch,
    )
    assert result.prediction is not None


def test_step_network_learns_and_predicts() -> None:
    state = create_network_state((2, 1))
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(raw_input="alpha beta gamma", top_down_feedback="Evaluate"),
            call_llm_mock_batch,
        )
        state = result.state
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(
                raw_input="alpha beta gamma",
                top_down_feedback="Error: Target: class_a",
            ),
            call_llm_mock_batch,
        )
        state = result.state
    working_state = reset_network_traces(state)
    for _ in range(2):
        result = step_network(
            working_state,
            NetworkStepInput(raw_input="gamma alpha beta", top_down_feedback="Evaluate"),
            call_llm_mock_batch,
        )
        working_state = result.state
    assert result.prediction == "class_a"


def test_step_network_continual_learning() -> None:
    state = create_network_state((2, 1))
    for label, features in (
        ("class_a", "alpha beta gamma"),
        ("class_b", "delta epsilon zeta"),
    ):
        for _ in range(2):
            result = step_network(
                state,
                NetworkStepInput(raw_input=features, top_down_feedback=f"Error: Target: {label}"),
                call_llm_mock_batch,
            )
            state = result.state
    labels = {
        item.output_text
        for item in parse_state_text(state.layers[-1].neurons[0].state).memories
        if item.output_text
    }
    assert labels == {"class_a", "class_b"}


def test_step_network_survives_neuron_removal() -> None:
    state = create_network_state((2, 1))
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(
                raw_input="alpha beta gamma",
                top_down_feedback="Error: Target: class_a",
            ),
            call_llm_mock_batch,
        )
        state = result.state
    reduced_layer = LayerState(neurons=state.layers[0].neurons[1:])
    reduced_state = NetworkState(
        layers=(reduced_layer,) + state.layers[1:],
        activations=(state.activations[0][1:],) + state.activations[1:],
        feedbacks=(state.feedbacks[0][1:],) + state.feedbacks[1:],
    )
    result = step_network(
        reset_network_traces(reduced_state),
        NetworkStepInput(raw_input="gamma alpha beta", top_down_feedback="Evaluate"),
        call_llm_mock_batch,
    )
    assert result.prediction in {"class_a", "unknown", "alpha+beta+gamma"}


def test_collect_all_requests() -> None:
    state = create_network_state((2, 1))
    requests = collect_all_requests(
        state,
        NetworkStepInput(raw_input="alpha beta gamma", top_down_feedback="Evaluate"),
    )
    assert len(requests) == 3
    assert requests[0].layer_index == 0
    assert requests[2].layer_index == 1
    assert all(request.prompt for request in requests)


def test_apply_all_results() -> None:
    state = create_network_state((2, 1))
    requests = collect_all_requests(
        state,
        NetworkStepInput(raw_input="alpha beta gamma", top_down_feedback="Evaluate"),
    )
    results = tuple(
        NeuronCallResult(request=request, response=call_llm_mock(request.prompt))
        for request in requests
    )
    step_result = apply_all_results(state, results)
    assert len(step_result.state.layers) == 2
    assert len(step_result.state.layers[0].neurons) == 2
    assert step_result.prediction is not None
