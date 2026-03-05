from __future__ import annotations

from continual_learning.llm import (
    apply_existing_rules,
    call_llm_mock,
    extract_prompt_field,
    extract_prompt_section,
    learn_new_rule,
    response_from_structured_output,
)
from continual_learning.neuron import build_neuron_prompt
from continual_learning.protocol import canonicalize_signal
from continual_learning.state import (
    MemoryEntryModel,
    StructuredNeuronResponseModel,
    StructuredNeuronStateModel,
    parse_state_text,
)
from continual_learning.types import NeuronResponse, NeuronState, NeuronStepInput


def test_extract_prompt_section() -> None:
    prompt = "CURRENT STATE (Persistent Memory JSON):\nsome rules\nINPUTS:\nstuff"
    result = extract_prompt_section(prompt, "CURRENT STATE (Persistent Memory JSON):", "INPUTS:")
    assert result == "some rules"


def test_extract_prompt_section_empty() -> None:
    result = extract_prompt_section("no match here", "START:", "END:")
    assert result == ""


def test_extract_prompt_field() -> None:
    prompt = "Bottom-up context: alpha beta gamma\nTop-down feedback: Evaluate"
    assert extract_prompt_field(prompt, "Bottom-up context") == "alpha beta gamma"
    assert extract_prompt_field(prompt, "Top-down feedback") == "Evaluate"


def test_extract_prompt_field_missing() -> None:
    assert extract_prompt_field("nothing here", "Missing field") == ""


def test_learn_new_rule_appends_memory() -> None:
    response = learn_new_rule("", "alpha beta gamma", "Error: Target: class_a")
    state = parse_state_text(response.new_state)
    learned = tuple(memory for memory in state.memories if memory.output_text == "class_a")
    assert learned[0].input_tokens == ("alpha", "beta", "gamma")
    assert response.activation_up == "class_a"


def test_learn_new_rule_accumulates_weight_for_duplicate() -> None:
    existing = learn_new_rule("", "alpha beta gamma", "Error: Target: class_a").new_state
    response = learn_new_rule(existing, "gamma alpha beta", "Error: Target: class_a")
    state = parse_state_text(response.new_state)
    learned = tuple(memory for memory in state.memories if memory.output_text == "class_a")
    assert len(learned) == 1
    assert learned[0].weight == 2


def test_learn_new_rule_preserves_old_memories() -> None:
    existing = learn_new_rule("", "alpha beta gamma", "Error: Target: class_a").new_state
    response = learn_new_rule(existing, "delta epsilon zeta", "Error: Target: class_b")
    labels = {item.output_text for item in parse_state_text(response.new_state).memories if item.output_text}
    assert labels == {"class_a", "class_b"}


def test_apply_existing_rules_matches_label_memory() -> None:
    state = learn_new_rule("", "alpha beta gamma", "Error: Target: class_a").new_state
    response = apply_existing_rules(state, "gamma alpha beta")
    assert response.activation_up == "class_a"


def test_apply_existing_rules_no_match_returns_token_projection() -> None:
    response = apply_existing_rules("", "alpha beta gamma")
    assert response.activation_up == "alpha+beta+gamma"


def test_apply_existing_rules_none_input() -> None:
    response = apply_existing_rules("", "None")
    assert response.activation_up == "unknown"


def test_call_llm_mock_learning() -> None:
    neuron = NeuronState(name="L1_N0")
    step_input = NeuronStepInput(
        bottom_up="alpha beta gamma",
        top_down="Error: Target: class_a",
    )
    prompt = build_neuron_prompt(neuron, step_input)
    response = call_llm_mock(prompt)
    assert response.activation_up == "class_a"
    learned = tuple(memory for memory in parse_state_text(response.new_state).memories if memory.output_text)
    assert learned[0].output_text == "class_a"


def test_call_llm_mock_inference() -> None:
    neuron = NeuronState(name="L1_N0", state=learn_new_rule(
        "", "alpha beta gamma", "Error: Target: class_a",
    ).new_state)
    step_input = NeuronStepInput(
        bottom_up="gamma alpha beta",
        top_down="Evaluate",
    )
    prompt = build_neuron_prompt(neuron, step_input)
    response = call_llm_mock(prompt)
    assert response.activation_up == "class_a"


def test_canonicalize_signal_normalizes_order_only() -> None:
    assert canonicalize_signal("gamma alpha beta alpha") == "alpha+beta+gamma"


def test_response_from_structured_output_serializes_memory_bank() -> None:
    parsed = StructuredNeuronResponseModel(
        new_state=StructuredNeuronStateModel(
            summary_tokens=("gamma", "alpha"),
            memories=(
                MemoryEntryModel(
                    input_tokens=("gamma", "alpha"),
                    output_text="class_a",
                    weight=3,
                ),
                MemoryEntryModel(
                    input_tokens=("gamma", "alpha"),
                    output_text="",
                    weight=2,
                ),
            ),
        ),
        activation_up="class_a",
        feedback_down="alpha+gamma",
    )
    normalized = response_from_structured_output(
        previous_state_text='{"memories":[],"summary_tokens":[],"version":1}',
        parsed=parsed,
        bottom_up="gamma alpha",
        top_down="Error: Target: class_a",
        allow_state_update=True,
    )
    state = parse_state_text(normalized.new_state)
    learned = tuple(memory for memory in state.memories if memory.output_text == "class_a")
    assert learned[0].input_tokens == ("alpha", "gamma")
    assert normalized.activation_up == "class_a"
    assert normalized.feedback_down == "alpha+gamma"


def test_response_from_structured_output_strips_unsupervised_label_memories() -> None:
    previous_state = learn_new_rule("", "alpha beta gamma", "Error: Target: class_a").new_state
    parsed = StructuredNeuronResponseModel(
        new_state=StructuredNeuronStateModel(
            summary_tokens=("gamma", "alpha"),
            memories=(
                MemoryEntryModel(
                    input_tokens=("gamma", "alpha"),
                    output_text="alpha+gamma",
                    weight=3,
                ),
            ),
        ),
        activation_up="alpha+gamma",
        feedback_down="none",
    )
    normalized = response_from_structured_output(
        previous_state_text=previous_state,
        parsed=parsed,
        bottom_up="gamma alpha",
        top_down="Evaluate",
        allow_state_update=True,
    )
    state = parse_state_text(normalized.new_state)
    labels = {item.output_text for item in state.memories if item.output_text}
    traces = tuple(item for item in state.memories if not item.output_text)
    assert labels == {"class_a"}
    assert traces[0].input_tokens == ("alpha", "gamma")


def test_parse_state_text_splits_joined_token_items() -> None:
    state = parse_state_text(
        (
            '{"version":1,"summary_tokens":["alpha+beta"],'
            '"memories":[{"input_tokens":["alpha+beta"],"output_text":"class_a","weight":1}]}'
        ),
    )
    assert state.summary_tokens == ("alpha", "beta")
    assert state.memories[0].input_tokens == ("alpha", "beta")
