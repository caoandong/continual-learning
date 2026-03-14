from __future__ import annotations

from dataclasses import replace

from continual_learning.constants import EMPTY_SIGNAL
from continual_learning.network import (
    build_layer_bottom_up,
    build_layer_top_down,
    create_network_state,
    reset_network_traces,
    step_network,
)
from continual_learning.types import NetworkStepInput, NeuronResponse

from tests.fakes import simple_learning_batch


def test_create_network_state_initializes_buffers() -> None:
    state = create_network_state((2, 1))
    assert state.latent_activations == ((EMPTY_SIGNAL, EMPTY_SIGNAL), (EMPTY_SIGNAL,))
    assert state.task_outputs == ((EMPTY_SIGNAL, EMPTY_SIGNAL), (EMPTY_SIGNAL,))
    assert state.latent_feedbacks == ((EMPTY_SIGNAL, EMPTY_SIGNAL), (EMPTY_SIGNAL,))


def test_create_network_state_uses_distinct_random_states() -> None:
    state = create_network_state((1, 1))
    states = tuple(neuron.state_text for layer in state.layers for neuron in layer.neurons)
    assert len(set(states)) == len(states)


def test_build_layer_bottom_up_routes_raw_input_to_layer_zero() -> None:
    state = create_network_state((1, 1))
    assert build_layer_bottom_up(state, 0, "raw pixels") == "raw pixels"


def test_build_layer_bottom_up_routes_previous_activations_upward() -> None:
    state = create_network_state((2, 1))
    state = replace(state, latent_activations=(("alpha", "beta"), (EMPTY_SIGNAL,)))
    assert build_layer_bottom_up(state, 1, "ignored") == "alpha | beta"


def test_build_layer_top_down_routes_external_feedback_to_top_layer() -> None:
    state = create_network_state((1, 1))
    assert build_layer_top_down(state, 1, "teacher") == "teacher"


def test_build_layer_top_down_routes_next_layer_feedback_downward() -> None:
    state = create_network_state((1, 1))
    state = replace(state, latent_feedbacks=((EMPTY_SIGNAL,), ("feedback",)))
    assert build_layer_top_down(state, 0, "ignored") == "feedback"


def test_read_only_step_preserves_state_when_caller_respects_mode() -> None:
    state = create_network_state((1, 1))
    before = tuple(neuron.state_text for layer in state.layers for neuron in layer.neurons)
    result = step_network(
        state,
        NetworkStepInput(
            raw_input="alpha beta",
            top_down_feedback=EMPTY_SIGNAL,
            allow_state_update=False,
        ),
        simple_learning_batch,
    )
    after = tuple(neuron.state_text for layer in result.state.layers for neuron in layer.neurons)
    assert before == after
    assert result.state.layers[0].neurons[0].last_latent == "alpha beta"


def test_write_step_replaces_state_only_with_returned_payload() -> None:
    def fixed_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
        return tuple(
            NeuronResponse(
                state_text="rewritten",
                latent_up="latent",
                task_up="task",
                latent_down="down",
            )
            for _ in prompts
        )

    result = step_network(
        create_network_state((1, 1)),
        NetworkStepInput(
            raw_input="alpha beta",
            teaching_signal="class_a",
            allow_state_update=True,
        ),
        fixed_batch,
    )
    assert all(neuron.state_text == "rewritten" for layer in result.state.layers for neuron in layer.neurons)
    assert result.latent_output == "latent"
    assert result.task_output == "task"


def test_batch_caller_runs_once_per_network_step() -> None:
    calls: list[tuple[str, ...]] = []

    def recording_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
        calls.append(prompts)
        return tuple(
            NeuronResponse(
                state_text="",
                latent_up=EMPTY_SIGNAL,
                task_up=EMPTY_SIGNAL,
                latent_down=EMPTY_SIGNAL,
            )
            for _ in prompts
        )

    step_network(
        create_network_state((2, 1)),
        NetworkStepInput(raw_input="alpha"),
        recording_batch,
    )
    assert len(calls) == 1
    assert len(calls[0]) == 3


def test_reset_network_traces_clears_transient_buffers() -> None:
    result = step_network(
        create_network_state((1, 1)),
        NetworkStepInput(raw_input="alpha beta"),
        simple_learning_batch,
    )
    reset = reset_network_traces(result.state)
    assert reset.latent_activations == ((EMPTY_SIGNAL,), (EMPTY_SIGNAL,))
    assert reset.task_outputs == ((EMPTY_SIGNAL,), (EMPTY_SIGNAL,))
    assert reset.latent_feedbacks == ((EMPTY_SIGNAL,), (EMPTY_SIGNAL,))
    assert all(neuron.last_latent == EMPTY_SIGNAL for layer in reset.layers for neuron in layer.neurons)
