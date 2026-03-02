from __future__ import annotations

from continual_learning.neuron import apply_neuron_response, build_neuron_prompt
from continual_learning.types import NeuronResponse, NeuronState, NeuronStepInput


def test_build_neuron_prompt_contains_name() -> None:
    neuron = NeuronState(name="L0_N1")
    step_input = NeuronStepInput(bottom_up="some input", top_down="some feedback")
    prompt = build_neuron_prompt(neuron, step_input)
    assert "L0_N1" in prompt


def test_build_neuron_prompt_contains_state() -> None:
    neuron = NeuronState(name="test", state="[If 'x' -> '1']")
    step_input = NeuronStepInput(bottom_up="x", top_down="y")
    prompt = build_neuron_prompt(neuron, step_input)
    assert "[If 'x' -> '1']" in prompt


def test_build_neuron_prompt_contains_inputs() -> None:
    neuron = NeuronState(name="test")
    step_input = NeuronStepInput(bottom_up="my_input", top_down="my_feedback")
    prompt = build_neuron_prompt(neuron, step_input)
    assert "my_input" in prompt
    assert "my_feedback" in prompt


def test_build_neuron_prompt_contains_last_output() -> None:
    neuron = NeuronState(name="test", last_output="prev_value")
    step_input = NeuronStepInput(bottom_up="x", top_down="y")
    prompt = build_neuron_prompt(neuron, step_input)
    assert "prev_value" in prompt


def test_apply_neuron_response_updates_state() -> None:
    neuron = NeuronState(name="L0_N0", state="old state")
    response = NeuronResponse(
        new_state="new state",
        activation_up="signal",
        feedback_down="feedback",
    )
    updated = apply_neuron_response(neuron, response)
    assert updated.state == "new state"
    assert updated.last_output == "signal"
    assert updated.name == "L0_N0"


def test_apply_neuron_response_preserves_name() -> None:
    neuron = NeuronState(name="important_name")
    response = NeuronResponse(new_state="s", activation_up="a", feedback_down="f")
    updated = apply_neuron_response(neuron, response)
    assert updated.name == "important_name"


def test_neuron_state_is_frozen() -> None:
    neuron = NeuronState(name="test")
    try:
        neuron.state = "mutated"  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass
