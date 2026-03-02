from __future__ import annotations

import functools
import json
import logging
import re

from continual_learning.constants import (
    DEFAULT_TEMPERATURE,
    EMPTY_SIGNAL,
    NEURON_SYSTEM_PROMPT,
)
from continual_learning.types import ExperimentOptions, LlmCaller, NeuronResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt parsing helpers
# ---------------------------------------------------------------------------

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
    pattern = re.escape(field_name) + r":\s*(.*)"
    match = re.search(pattern, prompt)
    result = match.group(1).strip() if match else ""
    logger.debug("[llm] extract_prompt_field '%s' -> %s", field_name, result)
    return result


# ---------------------------------------------------------------------------
# Mock LLM — deterministic rule engine
# ---------------------------------------------------------------------------

def learn_new_rule(state: str, bottom_up: str, top_down: str) -> NeuronResponse:
    target = top_down.split("Target:")[1].strip()
    rule = f"[If '{bottom_up}' -> '{target}']"
    new_state = state
    if rule not in new_state:
        new_state = (new_state + " " + rule).strip()
        logger.debug(
            "[llm] learn_new_rule APPENDED rule=%s\n"
            "  old_state=%s\n"
            "  new_state=%s",
            rule, state, new_state,
        )
    else:
        logger.debug("[llm] learn_new_rule DUPLICATE rule=%s (no change)", rule)
    response = NeuronResponse(
        new_state=new_state,
        activation_up=target,
        feedback_down=f"Error: Needs features for {target}",
    )
    logger.debug(
        "[llm] learn_new_rule RESPONSE\n"
        "  new_state=%s\n"
        "  activation_up=%s\n"
        "  feedback_down=%s",
        response.new_state, response.activation_up, response.feedback_down,
    )
    return response


def apply_existing_rules(state: str, bottom_up: str) -> NeuronResponse:
    match = re.search(r"\[If '" + re.escape(bottom_up) + r"' -> '(.*?)'\]", state)
    if match:
        activation = match.group(1)
        logger.debug(
            "[llm] apply_existing_rules MATCHED '%s' -> '%s' in state=%s",
            bottom_up, activation, state,
        )
    elif bottom_up != EMPTY_SIGNAL:
        activation = f"Features({bottom_up})"
        logger.debug(
            "[llm] apply_existing_rules NO MATCH for '%s' -> Features fallback=%s",
            bottom_up, activation,
        )
    else:
        activation = "unknown"
        logger.debug("[llm] apply_existing_rules EMPTY_SIGNAL input -> unknown")
    response = NeuronResponse(
        new_state=state,
        activation_up=activation,
        feedback_down="none",
    )
    logger.debug(
        "[llm] apply_existing_rules RESPONSE\n"
        "  new_state=%s\n"
        "  activation_up=%s\n"
        "  feedback_down=%s",
        response.new_state, response.activation_up, response.feedback_down,
    )
    return response


def call_llm_mock(prompt: str) -> NeuronResponse:
    logger.debug("[llm] call_llm_mock FULL PROMPT:\n%s", prompt)
    state = extract_prompt_section(
        prompt, "CURRENT STATE (Tokens as Weights):", "INPUTS:",
    )
    bottom_up = extract_prompt_field(prompt, "Bottom-up context")
    top_down = extract_prompt_field(prompt, "Top-down feedback")

    logger.debug(
        "[llm] call_llm_mock PARSED\n"
        "  state=%s\n"
        "  bottom_up=%s\n"
        "  top_down=%s",
        state, bottom_up, top_down,
    )

    if "Error" in top_down and "Target:" in top_down:
        logger.debug("[llm] call_llm_mock BRANCH -> learn_new_rule")
        return learn_new_rule(state, bottom_up, top_down)
    logger.debug("[llm] call_llm_mock BRANCH -> apply_existing_rules")
    return apply_existing_rules(state, bottom_up)


# ---------------------------------------------------------------------------
# Real LLM via litellm
# ---------------------------------------------------------------------------

def parse_llm_json(content: str) -> NeuronResponse:
    logger.debug("[llm] parse_llm_json raw content:\n%s", content)
    data = json.loads(content)
    response = NeuronResponse(
        new_state=data.get("new_state", ""),
        activation_up=data.get("activation_up", "unknown"),
        feedback_down=data.get("feedback_down", "none"),
    )
    logger.debug(
        "[llm] parse_llm_json PARSED\n"
        "  new_state=%s\n"
        "  activation_up=%s\n"
        "  feedback_down=%s",
        response.new_state, response.activation_up, response.feedback_down,
    )
    return response


def call_llm_litellm(prompt: str, *, model: str) -> NeuronResponse:
    import litellm  # noqa: PLC0415 — deferred to avoid import cost when using mock

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
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content  # type: ignore[union-attr]
    logger.debug("[llm] call_llm_litellm RAW RESPONSE:\n%s", content)
    return parse_llm_json(content)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_caller(options: ExperimentOptions) -> LlmCaller:
    if options.use_mock:
        logger.info("[llm] Using mock LLM")
        return call_llm_mock
    logger.info("[llm] Using litellm with model=%s", options.model)
    return functools.partial(call_llm_litellm, model=options.model)
