from __future__ import annotations

import logging
from dataclasses import replace

from continual_learning.constants import (
    DEFAULT_SIGNAL_TOKEN_BUDGET,
    DEFAULT_STATE_TOKEN_BUDGET,
    NEURON_PROMPT_TEMPLATE,
)
from continual_learning.types import NeuronResponse, NeuronState

logger = logging.getLogger(__name__)


def build_neuron_prompt(
    neuron: NeuronState,
    *,
    bottom_up: str,
    top_down: str,
    teaching_signal: str,
    allow_state_update: bool,
) -> str:
    prompt = NEURON_PROMPT_TEMPLATE.format(
        name=neuron.name,
        state_text=neuron.state_text,
        bottom_up=bottom_up,
        top_down=top_down,
        teaching_signal=teaching_signal,
        state_update_mode="write" if allow_state_update else "read_only",
        last_latent=neuron.last_latent,
        signal_token_budget=DEFAULT_SIGNAL_TOKEN_BUDGET,
        state_token_budget=DEFAULT_STATE_TOKEN_BUDGET,
    )
    logger.debug(
        "[neuron] build_neuron_prompt %s bottom_up=%s top_down=%s teaching_signal=%s allow_state_update=%s last_latent=%s",
        neuron.name,
        bottom_up,
        top_down,
        teaching_signal,
        allow_state_update,
        neuron.last_latent,
    )
    return prompt


def apply_neuron_response(neuron: NeuronState, response: NeuronResponse) -> NeuronState:
    updated = replace(
        neuron,
        state_text=response.state_text,
        last_latent=response.latent_up,
    )
    logger.debug(
        "[neuron] apply_neuron_response %s state=%s latent_up=%s task_up=%s latent_down=%s",
        neuron.name,
        updated.state_text,
        response.latent_up,
        response.task_up,
        response.latent_down,
    )
    return updated
