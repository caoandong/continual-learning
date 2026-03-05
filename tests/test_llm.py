from __future__ import annotations

from continual_learning.llm import (
    apply_existing_rules,
    call_llm_mock,
    extract_prompt_field,
    extract_prompt_section,
    learn_new_rule,
)
from continual_learning.neuron import build_neuron_prompt
from continual_learning.types import NeuronState, NeuronStepInput


def test_extract_prompt_section() -> None:
    prompt = "CURRENT STATE (Tokens as Weights):\nsome rules\nINPUTS:\nstuff"
    result = extract_prompt_section(prompt, "CURRENT STATE (Tokens as Weights):", "INPUTS:")
    assert result == "some rules"


def test_extract_prompt_section_empty() -> None:
    result = extract_prompt_section("no match here", "START:", "END:")
    assert result == ""


def test_extract_prompt_field() -> None:
    prompt = "Bottom-up context: Round closed loop\nTop-down feedback: Evaluate"
    assert extract_prompt_field(prompt, "Bottom-up context") == "Round closed loop"
    assert extract_prompt_field(prompt, "Top-down feedback") == "Evaluate"


def test_extract_prompt_field_missing() -> None:
    assert extract_prompt_field("nothing here", "Missing field") == ""


def test_learn_new_rule_appends() -> None:
    response = learn_new_rule("", "Round closed loop", "Error: Target: 0")
    assert "[If 'Round closed loop' -> '0']" in response.new_state
    assert response.activation_up == "0"


def test_learn_new_rule_no_duplicate() -> None:
    existing = "[If 'Round closed loop' -> '0']"
    response = learn_new_rule(existing, "Round closed loop", "Error: Target: 0")
    assert response.new_state.count("[If 'Round closed loop' -> '0']") == 1


def test_learn_new_rule_preserves_old() -> None:
    existing = "[If 'Round closed loop' -> '0']"
    response = learn_new_rule(existing, "Straight vertical line", "Error: Target: 1")
    assert "[If 'Round closed loop' -> '0']" in response.new_state
    assert "[If 'Straight vertical line' -> '1']" in response.new_state


def test_apply_existing_rules_matches() -> None:
    state = "[If 'Round closed loop' -> '0']"
    response = apply_existing_rules(state, "Round closed loop")
    assert response.activation_up == "0"


def test_apply_existing_rules_no_match() -> None:
    response = apply_existing_rules("", "Round closed loop")
    assert response.activation_up == "round"


def test_apply_existing_rules_none_input() -> None:
    response = apply_existing_rules("", "None")
    assert response.activation_up == "unknown"


def test_call_llm_mock_learning() -> None:
    neuron = NeuronState(name="L1_N0")
    step_input = NeuronStepInput(
        bottom_up="Round closed loop",
        top_down="Error: Target: 0",
    )
    prompt = build_neuron_prompt(neuron, step_input)
    response = call_llm_mock(prompt)
    assert response.activation_up == "0"
    assert "[If 'Round closed loop' -> '0']" in response.new_state


def test_call_llm_mock_inference() -> None:
    neuron = NeuronState(
        name="L1_N0",
        state="[If 'Round closed loop' -> '0']",
    )
    step_input = NeuronStepInput(
        bottom_up="Round closed loop",
        top_down="Evaluate",
    )
    prompt = build_neuron_prompt(neuron, step_input)
    response = call_llm_mock(prompt)
    assert response.activation_up == "0"
