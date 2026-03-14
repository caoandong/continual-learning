from __future__ import annotations

from continual_learning.constants import EMPTY_SIGNAL
from continual_learning.experiment import evaluate_sample, train_on_sample
from continual_learning.network import create_network_state
from continual_learning.types import NeuronResponse

from tests.fakes import extract_prompt_field, extract_prompt_section, simple_learning_batch


def test_train_and_evaluate_single_sample() -> None:
    state = create_network_state((1, 1))
    state = train_on_sample(state, "alpha beta gamma", "class_a", simple_learning_batch)
    _, readout = evaluate_sample(state, "alpha beta gamma", simple_learning_batch)
    assert readout.latent_output == "alpha beta gamma"
    assert readout.task_output == "class_a"


def test_train_on_sample_supports_incremental_learning() -> None:
    state = create_network_state((1, 1))
    state = train_on_sample(state, "alpha beta gamma", "class_a", simple_learning_batch)
    state = train_on_sample(state, "delta epsilon zeta", "class_b", simple_learning_batch)
    _, readout_a = evaluate_sample(state, "alpha beta gamma", simple_learning_batch)
    _, readout_b = evaluate_sample(state, "delta epsilon zeta", simple_learning_batch)
    assert readout_a.task_output == "class_a"
    assert readout_b.task_output == "class_b"


def test_train_and_evaluate_work_with_one_neuron_per_layer() -> None:
    state = create_network_state((1, 1))
    state = train_on_sample(state, "raw raw raw", "label", simple_learning_batch)
    _, readout = evaluate_sample(state, "raw raw raw", simple_learning_batch)
    assert readout.task_output == "label"


def test_train_and_evaluate_chunked_input_sequences() -> None:
    sample = " ".join(f"token{i}" for i in range(96))
    state = create_network_state((1, 1))
    state = train_on_sample(state, sample, "long_label", simple_learning_batch)
    _, readout = evaluate_sample(state, sample, simple_learning_batch)
    assert readout.task_output == "long_label"


def test_cold_start_can_emit_latent_without_task_output() -> None:
    state = create_network_state((1, 1))
    _, readout = evaluate_sample(state, "alpha beta gamma", simple_learning_batch)
    assert readout.latent_output == "alpha beta gamma"
    assert readout.task_output == EMPTY_SIGNAL


def test_evaluate_sample_allows_read_only_settling() -> None:
    def delayed_retrieval_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
        responses: list[NeuronResponse] = []
        for prompt in prompts:
            state = extract_prompt_section(prompt, "CURRENT STATE:", "INPUTS:")
            teaching_signal = extract_prompt_field(prompt, "Teaching signal")
            mode = extract_prompt_field(prompt, "State update mode")
            last_output = extract_prompt_field(prompt, "Last latent output")
            if mode == "write" and teaching_signal not in ("", EMPTY_SIGNAL):
                responses.append(
                    NeuronResponse(
                        state_text=teaching_signal,
                        latent_up="summary",
                        task_up=teaching_signal,
                        latent_down=EMPTY_SIGNAL,
                    ),
                )
                continue
            if state not in ("", EMPTY_SIGNAL):
                task_output = EMPTY_SIGNAL if last_output in ("", EMPTY_SIGNAL) else state
            else:
                task_output = EMPTY_SIGNAL
            responses.append(
                NeuronResponse(
                    state_text=state,
                    latent_up="summary",
                    task_up=task_output,
                    latent_down=EMPTY_SIGNAL,
                ),
            )
        return tuple(responses)

    state = create_network_state((1, 1))
    state = train_on_sample(state, "alpha beta gamma", "class_a", delayed_retrieval_batch)
    _, readout = evaluate_sample(state, "alpha beta gamma", delayed_retrieval_batch)
    assert readout.task_output == "class_a"


def test_train_on_sample_retries_until_reset_prediction_matches_target() -> None:
    def retry_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
        responses: list[NeuronResponse] = []
        for prompt in prompts:
            state = extract_prompt_section(prompt, "CURRENT STATE:", "INPUTS:")
            teaching_signal = extract_prompt_field(prompt, "Teaching signal")
            mode = extract_prompt_field(prompt, "State update mode")
            last_output = extract_prompt_field(prompt, "Last latent output")
            if mode == "write" and teaching_signal not in ("", EMPTY_SIGNAL):
                if state == f"pending:{teaching_signal}":
                    state_text = f"bound:{teaching_signal}"
                elif state == f"bound:{teaching_signal}":
                    state_text = state
                else:
                    state_text = f"pending:{teaching_signal}"
                responses.append(
                    NeuronResponse(
                        state_text=state_text,
                        latent_up="summary",
                        task_up=teaching_signal,
                        latent_down=EMPTY_SIGNAL,
                    ),
                )
                continue
            if state.startswith("bound:"):
                label = state.split(":", 1)[1]
                task_output = EMPTY_SIGNAL if last_output in ("", EMPTY_SIGNAL) else label
            else:
                task_output = EMPTY_SIGNAL
            responses.append(
                NeuronResponse(
                    state_text=state,
                    latent_up="summary",
                    task_up=task_output,
                    latent_down=EMPTY_SIGNAL,
                ),
            )
        return tuple(responses)

    state = create_network_state((1, 1))
    state = train_on_sample(state, "alpha beta gamma", "class_a", retry_batch)
    _, readout = evaluate_sample(state, "alpha beta gamma", retry_batch)
    assert readout.task_output == "class_a"
