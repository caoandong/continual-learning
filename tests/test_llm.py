from __future__ import annotations

from continual_learning.constants import MAX_MEMORY_PATTERNS, NUMERIC_FINGERPRINT_SIZE
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
    ParsedNeuronState,
    StateEvolutionInput,
    StructuredNeuronResponseModel,
    StructuredNeuronStateModel,
    compress_memories,
    contrastive_training_tokens,
    choose_activation,
    evolve_state,
    input_evidence_tokens,
    label_prototype_tokens,
    memory_entry_from_tokens,
    memory_tokens_from_text,
    numeric_fingerprint_from_text,
    parse_state_text,
    projected_tokens,
    ranked_output_matches,
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


def test_extract_prompt_field_blank_does_not_capture_next_line() -> None:
    prompt = (
        "Bottom-up context: alpha beta gamma\n"
        "Sensory context: \n"
        "State update mode: read_only\n"
        "Your last output: None"
    )
    assert extract_prompt_field(prompt, "Sensory context") == ""
    assert extract_prompt_field(prompt, "State update mode") == "read_only"


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
    assert canonicalize_signal("gamma alpha beta alpha") == "gamma+alpha+beta"


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
        sensory_input="gamma alpha",
        allow_state_update=True,
    )
    state = parse_state_text(normalized.new_state)
    learned = tuple(memory for memory in state.memories if memory.output_text == "class_a")
    assert learned[0].input_tokens == ("gamma", "alpha")
    assert normalized.activation_up == "class_a"
    assert normalized.feedback_down == "gamma+alpha"


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
        sensory_input="gamma alpha",
        allow_state_update=True,
    )
    state = parse_state_text(normalized.new_state)
    labels = {item.output_text for item in state.memories if item.output_text}
    traces = tuple(item for item in state.memories if not item.output_text)
    assert labels == {"class_a"}
    assert traces[0].input_tokens == ("gamma", "alpha")


def test_response_from_structured_output_retains_labels_under_trace_pressure() -> None:
    previous_state = ""
    for label in ("0", "1", "2"):
        previous_state = learn_new_rule(
            previous_state,
            f"example {label}",
            f"Error: Target: {label}",
        ).new_state
    parsed = StructuredNeuronResponseModel(
        new_state=StructuredNeuronStateModel(
            memories=tuple(
                MemoryEntryModel(
                    input_tokens=(f"trace_{index}",),
                    output_text="",
                    weight=MAX_MEMORY_PATTERNS + index,
                )
                for index in range(MAX_MEMORY_PATTERNS + 2)
            ),
        ),
    )
    normalized = response_from_structured_output(
        previous_state_text=previous_state,
        parsed=parsed,
        bottom_up="trace stress",
        top_down="Evaluate",
        sensory_input="trace stress",
        allow_state_update=True,
    )
    state = parse_state_text(normalized.new_state)
    labels = {item.output_text for item in state.memories if item.output_text}
    assert labels == {"0", "1", "2"}


def test_parse_state_text_splits_joined_token_items() -> None:
    state = parse_state_text(
        (
            '{"version":1,"summary_tokens":["alpha+beta"],'
            '"memories":[{"input_tokens":["alpha+beta"],"output_text":"class_a","weight":1}]}'
        ),
    )
    assert state.summary_tokens == ("alpha", "beta")
    assert state.memories[0].input_tokens == ("alpha", "beta")


def test_learn_new_rule_compresses_long_raw_stream_into_signature_tokens() -> None:
    raw_stream = " ".join(str(index % 256) for index in range(128))
    response = learn_new_rule("", raw_stream, "Error: Target: class_a")
    state = parse_state_text(response.new_state)
    learned = tuple(memory for memory in state.memories if memory.output_text == "class_a")
    assert learned
    assert "len:128" in learned[0].input_tokens
    assert any(token.startswith("s4:") for token in learned[0].input_tokens)
    assert any(token.startswith("s16:") for token in learned[0].input_tokens)
    assert any(token.startswith("g:") for token in learned[0].input_tokens)
    assert any(token.startswith("dg:") for token in learned[0].input_tokens)
    assert any(token.startswith("r4:") for token in learned[0].input_tokens)
    assert any(token.startswith("s16:15:") for token in learned[0].input_tokens)
    assert any(token.startswith("s32:31:") for token in learned[0].input_tokens)
    assert any(token.startswith("n2:") for token in learned[0].input_tokens)
    assert any(token.startswith("a") and ":" in token for token in learned[0].input_tokens)
    assert learned[0].numeric_fingerprint


def test_ranked_output_matches_uses_numeric_fingerprint_similarity() -> None:
    class_a = memory_entry_from_tokens(
        ("shared",),
        "class_a",
        1,
        numeric_fingerprint=(1000, 0),
    )
    class_b = memory_entry_from_tokens(
        ("shared",),
        "class_b",
        1,
        numeric_fingerprint=(0, 1000),
    )
    assert class_a is not None
    assert class_b is not None
    state = ParsedNeuronState(
        summary_tokens=(),
        memories=(class_a, class_b),
        support_memories=(),
    )
    ranked = ranked_output_matches(
        state,
        ("shared",),
        numeric_fingerprint=(0, 1000),
    )
    assert ranked[0][0] == "class_b"


def test_numeric_fingerprint_from_text_is_length_bounded() -> None:
    raw_stream = " ".join(str(index) for index in range(256))
    fingerprint = numeric_fingerprint_from_text(raw_stream)
    assert fingerprint
    assert len(fingerprint) <= NUMERIC_FINGERPRINT_SIZE


def test_evolve_state_adds_support_memory_for_labeled_numeric_input() -> None:
    state = evolve_state(
        StateEvolutionInput(
            previous_state_text="",
            bottom_up="0 1 2 3",
            top_down="Error: Target: class_a",
            sensory_input="0 1 2 3",
        ),
    )
    assert state.support_memories
    assert state.support_memories[0].output_text == "class_a"
    assert state.support_memories[0].numeric_fingerprint


def test_choose_activation_prefers_support_memory_match() -> None:
    trained = evolve_state(
        StateEvolutionInput(
            previous_state_text="",
            bottom_up="0 1 2 3",
            top_down="Error: Target: class_a",
            sensory_input="0 1 2 3",
        ),
    )
    prediction = choose_activation(
        StateEvolutionInput(
            previous_state_text="",
            bottom_up="0 1 2 4",
            top_down="Evaluate",
            sensory_input="0 1 2 4",
        ),
        trained,
    )
    assert prediction == "class_a"


def test_input_evidence_tokens_preserves_distinct_channels() -> None:
    evidence = input_evidence_tokens(
        StateEvolutionInput(
            previous_state_text="",
            bottom_up="alpha beta gamma delta",
            top_down="theta+iota",
            sensory_input="theta iota kappa lambda",
        ),
    )
    assert any(token.startswith("sensory:") for token in evidence)
    assert any(token.startswith("feedback:") for token in evidence)
    assert any(token in {"alpha", "beta", "gamma", "delta"} for token in evidence)


def test_ranked_output_matches_uses_label_prototype_evidence() -> None:
    state = parse_state_text(
        StructuredNeuronStateModel(
            memories=(
                MemoryEntryModel(
                    input_tokens=("left", "tail"),
                    output_text="class_a",
                    weight=1,
                ),
                MemoryEntryModel(
                    input_tokens=("right", "tail"),
                    output_text="class_a",
                    weight=1,
                ),
                MemoryEntryModel(
                    input_tokens=("left", "right"),
                    output_text="class_b",
                    weight=1,
                ),
            ),
        ).model_dump_json(),
    )
    ranked = ranked_output_matches(state, ("left", "right", "tail"))
    assert ranked[0][0] == "class_a"


def test_compress_memories_reserves_one_slot_per_label() -> None:
    labeled = tuple(
        entry
        for entry in (
            memory_entry_from_tokens(memory_tokens_from_text(f"digit {label}"), label, 1)
            for label in ("0", "1", "2")
        )
        if entry is not None
    )
    unlabeled = tuple(
        entry
        for entry in (
            memory_entry_from_tokens(memory_tokens_from_text(f"trace {index}"), "", 10 + index)
            for index in range(MAX_MEMORY_PATTERNS + 3)
        )
        if entry is not None
    )
    compressed = compress_memories(labeled + unlabeled)
    labels = {memory.output_text for memory in compressed if memory.output_text}
    assert labels == {"0", "1", "2"}


def test_compress_memories_preserves_distinct_unlabeled_traces() -> None:
    unlabeled = tuple(
        entry
        for entry in (
            memory_entry_from_tokens(
                (
                    "common",
                    "shared",
                    "left_only",
                ),
                "",
                1,
            ),
            memory_entry_from_tokens(
                (
                    "common",
                    "shared",
                    "right_only",
                ),
                "",
                1,
            ),
        )
        if entry is not None
    )
    compressed = compress_memories(unlabeled)
    assert len(compressed) == 2


def test_projected_tokens_prefers_nearest_memory_over_global_average() -> None:
    state = parse_state_text(
        StructuredNeuronStateModel(
            memories=(
                MemoryEntryModel(
                    input_tokens=("common", "left_only", "left_detail"),
                    output_text="",
                    weight=1,
                ),
                MemoryEntryModel(
                    input_tokens=("common", "right_only", "right_detail"),
                    output_text="",
                    weight=8,
                ),
            ),
        ).model_dump_json(),
    )
    projected = projected_tokens(state, ("common", "left_only"))
    assert "left_only" in projected
    assert "right_only" not in projected[:2]


def test_projected_tokens_does_not_emit_feedback_channel_tokens() -> None:
    state = parse_state_text(
        StructuredNeuronStateModel(
            memories=(
                MemoryEntryModel(
                    input_tokens=("alpha", "feedback:trace_a"),
                    output_text="",
                    weight=2,
                ),
            ),
        ).model_dump_json(),
    )
    projected = projected_tokens(state, ("alpha", "feedback:trace_a"))
    assert projected == ("alpha",)


def test_response_from_structured_output_keeps_deterministic_learning_baseline() -> None:
    parsed = StructuredNeuronResponseModel(
        new_state=StructuredNeuronStateModel(
            summary_tokens=(),
            memories=(),
        ),
        activation_up="unknown",
        feedback_down="none",
    )
    normalized = response_from_structured_output(
        previous_state_text='{"memories":[],"summary_tokens":[],"version":1}',
        parsed=parsed,
        bottom_up="alpha beta gamma",
        top_down="Error: Target: class_a",
        sensory_input="alpha beta gamma",
        allow_state_update=True,
    )
    state = parse_state_text(normalized.new_state)
    labels = {item.output_text for item in state.memories if item.output_text}
    assert labels == {"class_a"}


def test_contrastive_training_tokens_drop_competing_prototype_tokens() -> None:
    state = parse_state_text(
        StructuredNeuronStateModel(
            memories=(
                MemoryEntryModel(
                    input_tokens=("shared", "target_only", "class_a_hint"),
                    output_text="class_a",
                    weight=1,
                ),
                MemoryEntryModel(
                    input_tokens=("shared", "wrong_only", "class_b_hint"),
                    output_text="class_b",
                    weight=1,
                ),
            ),
        ).model_dump_json(),
    )
    tokens = contrastive_training_tokens(
        state,
        output_text="class_a",
        predicted_output_text="class_b",
        input_tokens=("shared", "target_only", "wrong_only"),
    )
    assert "target_only" in tokens
    assert "wrong_only" not in tokens


def test_evolve_state_weakens_competing_label_memory_on_error() -> None:
    previous_state = StructuredNeuronStateModel(
        memories=(
            MemoryEntryModel(
                input_tokens=("shared", "wrong_only"),
                output_text="class_b",
                weight=2,
            ),
        ),
    ).model_dump_json()
    evolved = evolve_state(
        StateEvolutionInput(
            previous_state_text=previous_state,
            bottom_up="shared wrong_only target_only",
            top_down="Error: Predicted: class_b | Target: class_a",
            sensory_input="shared wrong_only target_only",
        ),
    )
    class_b_memories = tuple(memory for memory in evolved.memories if memory.output_text == "class_b")
    assert class_b_memories
    assert class_b_memories[0].weight == 1


def test_label_prototype_tokens_prefers_consensus_tokens() -> None:
    memories = tuple(
        memory
        for memory in (
            memory_entry_from_tokens(("shared", "a_only"), "class_a", 1),
            memory_entry_from_tokens(("shared", "b_only"), "class_a", 1),
            memory_entry_from_tokens(("shared", "c_only"), "class_a", 1),
        )
        if memory is not None
    )
    prototype = label_prototype_tokens(memories)
    assert prototype[0] == "shared"
