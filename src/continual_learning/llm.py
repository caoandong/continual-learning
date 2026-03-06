from __future__ import annotations

import functools
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from continual_learning.constants import DEFAULT_TEMPERATURE, NEURON_SYSTEM_PROMPT
from continual_learning.environment import load_environment_file
from continual_learning.state import (
    StateEvolutionInput,
    StructuredNeuronResponseModel,
    debug_state_summary,
    evolve_state,
    merge_state_updates,
    parse_state_text,
    parsed_state_from_model,
    readout_response,
    serialize_state,
    sanitize_state_update,
    state_response,
)
from continual_learning.types import (
    BatchLlmCaller,
    ExperimentOptions,
    LlmCaller,
    NeuronResponse,
)

logger = logging.getLogger(__name__)


def extract_prompt_section(prompt: str, section_start: str, section_end: str) -> str:
    pattern = re.escape(section_start) + r"\s*(.*?)\s*" + re.escape(section_end)
    match = re.search(pattern, prompt, re.DOTALL)
    result = match.group(1).strip() if match else ""
    logger.debug(
        "[llm] extract_prompt_section '%s'...'%s' -> %s",
        section_start, section_end, result,
    )
    return result


def extract_prompt_field(prompt: str, field_name: str) -> str:
    pattern = r"^" + re.escape(field_name) + r":[ \t]*(.*)$"
    match = re.search(pattern, prompt, re.MULTILINE)
    result = match.group(1).strip() if match else ""
    logger.debug("[llm] extract_prompt_field '%s' -> %s", field_name, result)
    return result


def response_from_state_evolution(input_data: StateEvolutionInput) -> NeuronResponse:
    new_state, activation_up, feedback_down = state_response(input_data)
    response = NeuronResponse(
        new_state=new_state,
        activation_up=activation_up,
        feedback_down=feedback_down,
    )
    logger.debug(
        "[llm] response_from_state_evolution\n"
        "  previous_state=%s\n"
        "  next_state=%s\n"
        "  activation_up=%s\n"
        "  feedback_down=%s",
        debug_state_summary(input_data.previous_state_text),
        debug_state_summary(response.new_state),
        response.activation_up,
        response.feedback_down,
    )
    return response


def response_from_state_readout(
    *,
    state_text: str,
    bottom_up: str,
    top_down: str,
    sensory_input: str = "",
    preferred_activation: str = "",
    preferred_feedback: str = "",
) -> NeuronResponse:
    state = parse_state_text(state_text)
    activation_up, feedback_down = readout_response(
        state=state,
        bottom_up=bottom_up,
        top_down=top_down,
        sensory_input=sensory_input,
        preferred_activation=preferred_activation,
        preferred_feedback=preferred_feedback,
    )
    return NeuronResponse(
        new_state=serialize_state(state),
        activation_up=activation_up,
        feedback_down=feedback_down,
    )


def learn_new_rule(state: str, bottom_up: str, top_down: str) -> NeuronResponse:
    return response_from_state_evolution(
        StateEvolutionInput(
            previous_state_text=state,
            bottom_up=bottom_up,
            top_down=top_down,
            sensory_input=bottom_up,
        ),
    )


def apply_existing_rules(state: str, bottom_up: str) -> NeuronResponse:
    return response_from_state_evolution(
        StateEvolutionInput(
            previous_state_text=state,
            bottom_up=bottom_up,
            top_down="Evaluate",
            sensory_input=bottom_up,
        ),
    )


def call_llm_mock(prompt: str) -> NeuronResponse:
    logger.debug("[llm] call_llm_mock FULL PROMPT:\n%s", prompt)
    state = extract_prompt_section(
        prompt, "CURRENT STATE (Persistent Memory JSON):", "INPUTS:",
    )
    bottom_up = extract_prompt_field(prompt, "Bottom-up context")
    top_down = extract_prompt_field(prompt, "Top-down feedback")
    sensory_input = extract_prompt_field(prompt, "Sensory context")
    state_update_mode = extract_prompt_field(prompt, "State update mode")
    logger.debug(
        "[llm] call_llm_mock PARSED\n"
        "  state=%s\n"
        "  bottom_up=%s\n"
        "  top_down=%s\n"
        "  sensory_input=%s\n"
        "  state_update_mode=%s",
        debug_state_summary(state), bottom_up, top_down, sensory_input, state_update_mode,
    )
    if state_update_mode == "read_only":
        return response_from_state_readout(
            state_text=state,
            bottom_up=bottom_up,
            top_down=top_down,
            sensory_input=sensory_input,
        )
    return response_from_state_evolution(
        StateEvolutionInput(
            previous_state_text=state,
            bottom_up=bottom_up,
            top_down=top_down,
            sensory_input=sensory_input,
        ),
    )


def response_from_structured_output(
    *,
    previous_state_text: str,
    parsed: StructuredNeuronResponseModel,
    bottom_up: str,
    top_down: str,
    sensory_input: str,
    allow_state_update: bool,
) -> NeuronResponse:
    previous_state = parse_state_text(previous_state_text)
    state = previous_state
    if allow_state_update:
        deterministic_state = evolve_state(
            StateEvolutionInput(
                previous_state_text=previous_state_text,
                bottom_up=bottom_up,
                top_down=top_down,
                sensory_input=sensory_input,
            ),
        )
        proposed_state = parsed_state_from_model(parsed.new_state)
        sanitized_state = sanitize_state_update(
            previous_state=previous_state,
            proposed_state=proposed_state,
            top_down=top_down,
        )
        state = merge_state_updates(deterministic_state, sanitized_state)
    activation_up, feedback_down = readout_response(
        state=state,
        bottom_up=bottom_up,
        top_down=top_down,
        sensory_input=sensory_input,
        preferred_activation=parsed.activation_up,
        preferred_feedback=parsed.feedback_down,
    )
    return NeuronResponse(
        new_state=serialize_state(state),
        activation_up=activation_up,
        feedback_down=feedback_down,
    )


def parse_structured_message(message: object) -> StructuredNeuronResponseModel:
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, StructuredNeuronResponseModel):
        return parsed

    content = getattr(message, "content", None)
    if isinstance(content, StructuredNeuronResponseModel):
        return content
    if isinstance(content, str):
        return StructuredNeuronResponseModel.model_validate_json(content)
    if isinstance(content, dict):
        return StructuredNeuronResponseModel.model_validate(content)
    raise TypeError(f"Unsupported LiteLLM message payload: {type(content)!r}")


def call_llm_litellm(prompt: str, *, model: str) -> NeuronResponse:
    import litellm  # noqa: PLC0415 — deferred to avoid import cost when using mock

    state = extract_prompt_section(
        prompt, "CURRENT STATE (Persistent Memory JSON):", "INPUTS:",
    )
    bottom_up = extract_prompt_field(prompt, "Bottom-up context")
    top_down = extract_prompt_field(prompt, "Top-down feedback")
    sensory_input = extract_prompt_field(prompt, "Sensory context")
    allow_state_update = extract_prompt_field(prompt, "State update mode") != "read_only"
    logger.debug(
        "[llm] call_llm_litellm model=%s FULL PROMPT:\n%s",
        model, prompt,
    )
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": NEURON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=DEFAULT_TEMPERATURE,
        response_format=StructuredNeuronResponseModel,
    )
    message = response.choices[0].message  # type: ignore[union-attr]
    parsed = parse_structured_message(message)
    logger.debug("[llm] call_llm_litellm PARSED RESPONSE:\n%s", parsed.model_dump_json())
    return response_from_structured_output(
        previous_state_text=state,
        parsed=parsed,
        bottom_up=bottom_up,
        top_down=top_down,
        sensory_input=sensory_input,
        allow_state_update=allow_state_update,
    )


def create_llm_caller(options: ExperimentOptions) -> LlmCaller:
    if options.use_mock:
        logger.info("[llm] Using mock LLM")
        return call_llm_mock
    load_environment_file()
    logger.info("[llm] Using litellm with model=%s", options.model)
    return functools.partial(call_llm_litellm, model=options.model)


def call_llm_batch_threaded(
    prompts: tuple[str, ...], call_llm: LlmCaller,
) -> tuple[NeuronResponse, ...]:
    if not prompts:
        return ()
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(call_llm, prompt) for prompt in prompts]
        return tuple(future.result() for future in futures)


def create_batch_llm_caller(options: ExperimentOptions) -> BatchLlmCaller:
    call_llm = create_llm_caller(options)
    return functools.partial(call_llm_batch_threaded, call_llm=call_llm)
