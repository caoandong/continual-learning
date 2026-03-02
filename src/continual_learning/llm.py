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
    return match.group(1).strip() if match else ""


def extract_prompt_field(prompt: str, field_name: str) -> str:
    pattern = re.escape(field_name) + r":\s*(.*)"
    match = re.search(pattern, prompt)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Mock LLM — deterministic rule engine
# ---------------------------------------------------------------------------

def learn_new_rule(state: str, bottom_up: str, top_down: str) -> NeuronResponse:
    target = top_down.split("Target:")[1].strip()
    rule = f"[If '{bottom_up}' -> '{target}']"
    new_state = state
    if rule not in new_state:
        new_state = (new_state + " " + rule).strip()
    return NeuronResponse(
        new_state=new_state,
        activation_up=target,
        feedback_down=f"Error: Needs features for {target}",
    )


def apply_existing_rules(state: str, bottom_up: str) -> NeuronResponse:
    match = re.search(r"\[If '" + re.escape(bottom_up) + r"' -> '(.*?)'\]", state)
    if match:
        activation = match.group(1)
    elif bottom_up != EMPTY_SIGNAL:
        activation = f"Features({bottom_up})"
    else:
        activation = "unknown"
    return NeuronResponse(
        new_state=state,
        activation_up=activation,
        feedback_down="none",
    )


def call_llm_mock(prompt: str) -> NeuronResponse:
    state = extract_prompt_section(
        prompt, "CURRENT STATE (Tokens as Weights):", "INPUTS:",
    )
    bottom_up = extract_prompt_field(prompt, "Bottom-up context")
    top_down = extract_prompt_field(prompt, "Top-down feedback")

    if "Error" in top_down and "Target:" in top_down:
        return learn_new_rule(state, bottom_up, top_down)
    return apply_existing_rules(state, bottom_up)


# ---------------------------------------------------------------------------
# Real LLM via litellm
# ---------------------------------------------------------------------------

def parse_llm_json(content: str) -> NeuronResponse:
    data = json.loads(content)
    return NeuronResponse(
        new_state=data.get("new_state", ""),
        activation_up=data.get("activation_up", "unknown"),
        feedback_down=data.get("feedback_down", "none"),
    )


def call_llm_litellm(prompt: str, *, model: str) -> NeuronResponse:
    import litellm  # noqa: PLC0415 — deferred to avoid import cost when using mock

    logger.debug("[llm] Calling %s", model)
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
