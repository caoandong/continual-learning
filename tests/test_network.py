from __future__ import annotations

from continual_learning.constants import EMPTY_SIGNAL
from continual_learning.llm import call_llm_mock
from continual_learning.network import (
    apply_all_results,
    build_layer_bottom_up,
    build_layer_top_down,
    collect_all_requests,
    create_network_state,
    step_network,
)
from continual_learning.types import (
    LayerState,
    NetworkState,
    NetworkStepInput,
    NeuronCallResult,
    NeuronResponse,
)


def call_llm_mock_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
    return tuple(call_llm_mock(p) for p in prompts)


def test_create_network_state_default() -> None:
    state = create_network_state((2, 1))
    assert len(state.layers) == 2
    assert len(state.layers[0].neurons) == 2
    assert len(state.layers[1].neurons) == 1
    assert state.layers[0].neurons[0].name == "L0_N0"
    assert state.layers[0].neurons[1].name == "L0_N1"
    assert state.layers[1].neurons[0].name == "L1_N0"


def test_create_network_state_buffers_initialized() -> None:
    state = create_network_state((3, 2))
    assert len(state.activations[0]) == 3
    assert len(state.activations[1]) == 2
    assert all(a == EMPTY_SIGNAL for a in state.activations[0])
    assert all(f == EMPTY_SIGNAL for f in state.feedbacks[1])


def test_build_layer_bottom_up_layer_zero() -> None:
    state = create_network_state((2, 1))
    result = build_layer_bottom_up(state, 0, "raw pixels")
    assert result == "raw pixels"


def test_build_layer_bottom_up_higher_layer() -> None:
    state = create_network_state((2, 1))
    result = build_layer_bottom_up(state, 1, "raw pixels")
    assert result == f"{EMPTY_SIGNAL} | {EMPTY_SIGNAL}"


def test_build_layer_top_down_top_layer() -> None:
    state = create_network_state((2, 1))
    result = build_layer_top_down(state, 1, "Evaluate")
    assert result == "Evaluate"


def test_build_layer_top_down_lower_layer() -> None:
    state = create_network_state((2, 1))
    result = build_layer_top_down(state, 0, "Evaluate")
    assert result == EMPTY_SIGNAL


def test_step_network_returns_prediction() -> None:
    state = create_network_state((2, 1))
    step_input = NetworkStepInput(raw_input="Round closed loop", top_down_feedback="Evaluate")
    result = step_network(state, step_input, call_llm_mock_batch)
    assert result.prediction is not None
    assert result.state is not None


def test_step_network_learns_and_predicts() -> None:
    state = create_network_state((2, 1))

    # Propagation ticks
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(raw_input="Round closed loop", top_down_feedback="Evaluate"),
            call_llm_mock_batch,
        )
        state = result.state

    # Error correction ticks
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(
                raw_input="Round closed loop",
                top_down_feedback="Error: Target: 0",
            ),
            call_llm_mock_batch,
        )
        state = result.state

    # Verify learned rule persists in output neuron
    output_neuron = state.layers[-1].neurons[0]
    assert "0" in output_neuron.state

    # Now test inference — should predict 0
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(raw_input="Round closed loop", top_down_feedback="Evaluate"),
            call_llm_mock_batch,
        )
        state = result.state
    assert result.prediction == "0"


def test_step_network_continual_learning() -> None:
    state = create_network_state((2, 1))

    # Teach digit 0
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(
                raw_input="Round closed loop",
                top_down_feedback="Error: Target: 0",
            ),
            call_llm_mock_batch,
        )
        state = result.state

    # Teach digit 1
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(
                raw_input="Straight vertical line",
                top_down_feedback="Error: Target: 1",
            ),
            call_llm_mock_batch,
        )
        state = result.state

    output_neuron = state.layers[-1].neurons[0]
    # Both rules should be present (no catastrophic forgetting)
    assert "'0'" in output_neuron.state
    assert "'1'" in output_neuron.state


def test_step_network_survives_neuron_removal() -> None:
    state = create_network_state((2, 1))

    # Train briefly
    for _ in range(2):
        result = step_network(
            state,
            NetworkStepInput(
                raw_input="Round closed loop",
                top_down_feedback="Error: Target: 0",
            ),
            call_llm_mock_batch,
        )
        state = result.state

    # Remove a neuron from layer 0
    reduced_layer = LayerState(neurons=state.layers[0].neurons[1:])
    reduced_state = NetworkState(
        layers=(reduced_layer,) + state.layers[1:],
        activations=(state.activations[0][1:],) + state.activations[1:],
        feedbacks=(state.feedbacks[0][1:],) + state.feedbacks[1:],
    )

    # Should not crash
    result = step_network(
        reduced_state,
        NetworkStepInput(raw_input="Straight vertical line", top_down_feedback="Evaluate"),
        call_llm_mock_batch,
    )
    assert result.prediction is not None


def test_collect_all_requests() -> None:
    state = create_network_state((2, 1))
    step_input = NetworkStepInput(raw_input="Round closed loop", top_down_feedback="Evaluate")
    requests = collect_all_requests(state, step_input)

    # 2 neurons in layer 0 + 1 neuron in layer 1 = 3 requests
    assert len(requests) == 3
    assert requests[0].layer_index == 0
    assert requests[0].neuron_index == 0
    assert requests[1].layer_index == 0
    assert requests[1].neuron_index == 1
    assert requests[2].layer_index == 1
    assert requests[2].neuron_index == 0
    # Each request has a non-empty prompt
    assert all(r.prompt for r in requests)


def test_apply_all_results() -> None:
    state = create_network_state((2, 1))
    step_input = NetworkStepInput(raw_input="Round closed loop", top_down_feedback="Evaluate")
    requests = collect_all_requests(state, step_input)

    # Build results by running mock on each prompt
    results = tuple(
        NeuronCallResult(request=req, response=call_llm_mock(req.prompt))
        for req in requests
    )
    step_result = apply_all_results(state, results)

    # Output network should have same structure
    assert len(step_result.state.layers) == 2
    assert len(step_result.state.layers[0].neurons) == 2
    assert len(step_result.state.layers[1].neurons) == 1
    assert step_result.prediction is not None
