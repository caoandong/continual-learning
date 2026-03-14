from __future__ import annotations

from continual_learning.neuron import apply_neuron_response, build_neuron_prompt
from continual_learning.types import NeuronResponse, NeuronState


def test_build_neuron_prompt_contains_minimal_fields_only() -> None:
    prompt = build_neuron_prompt(
        NeuronState(name="L0_N0", state_text="seed0 seed1", last_latent="old"),
        bottom_up="alpha beta",
        top_down="higher layer",
        teaching_signal="label",
        allow_state_update=False,
    )
    assert "CURRENT STATE:" in prompt
    assert "Bottom-up context: alpha beta" in prompt
    assert "Top-down feedback: higher layer" in prompt
    assert "Teaching signal: label" in prompt
    assert "State update mode: read_only" in prompt
    assert "Last latent output: old" in prompt


def test_build_neuron_prompt_excludes_old_special_case_fields() -> None:
    prompt = build_neuron_prompt(
        NeuronState(name="L0_N0"),
        bottom_up="raw input",
        top_down="feedback",
        teaching_signal="target",
        allow_state_update=True,
    )
    assert "Sensory context" not in prompt
    assert "numeric_fingerprint" not in prompt
    assert "memories" not in prompt


def test_apply_neuron_response_updates_state_and_last_output() -> None:
    updated = apply_neuron_response(
        NeuronState(name="L0_N0", state_text="before", last_latent="old"),
        NeuronResponse(
            state_text="after",
            latent_up="signal",
            task_up="task",
            latent_down="down",
        ),
    )
    assert updated.name == "L0_N0"
    assert updated.state_text == "after"
    assert updated.last_latent == "signal"
