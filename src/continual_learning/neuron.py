from __future__ import annotations

import logging
from dataclasses import replace

from continual_learning.constants import NEURON_PROMPT_TEMPLATE
from continual_learning.types import NeuronResponse, NeuronState, NeuronStepInput

logger = logging.getLogger(__name__)


def build_neuron_prompt(neuron: NeuronState, step_input: NeuronStepInput) -> str:
    prompt = NEURON_PROMPT_TEMPLATE.format(
        name=neuron.name,
        state=neuron.state,
        bottom_up=step_input.bottom_up,
        top_down=step_input.top_down,
        state_update_mode="write" if step_input.allow_state_update else "read_only",
        last_output=neuron.last_output,
    )
    logger.debug(
        "[neuron] build_neuron_prompt %s\n"
        "  inputs: bottom_up=%s | top_down=%s | allow_state_update=%s | last_output=%s\n"
        "  current_state=%s",
        neuron.name,
        step_input.bottom_up,
        step_input.top_down,
        step_input.allow_state_update,
        neuron.last_output,
        neuron.state,
    )
    return prompt


def apply_neuron_response(neuron: NeuronState, response: NeuronResponse) -> NeuronState:
    updated = replace(
        neuron,
        state=response.new_state,
        last_output=response.activation_up,
    )
    logger.debug(
        "[neuron] apply_neuron_response %s\n"
        "  BEFORE: state=%s | last_output=%s\n"
        "  AFTER:  state=%s | last_output=%s",
        neuron.name,
        neuron.state,
        neuron.last_output,
        updated.state,
        updated.last_output,
    )
    return updated
