from __future__ import annotations

from dataclasses import replace

from continual_learning.constants import NEURON_PROMPT_TEMPLATE
from continual_learning.types import NeuronResponse, NeuronState, NeuronStepInput


def build_neuron_prompt(neuron: NeuronState, step_input: NeuronStepInput) -> str:
    return NEURON_PROMPT_TEMPLATE.format(
        name=neuron.name,
        state=neuron.state,
        bottom_up=step_input.bottom_up,
        top_down=step_input.top_down,
        last_output=neuron.last_output,
    )


def apply_neuron_response(neuron: NeuronState, response: NeuronResponse) -> NeuronState:
    return replace(
        neuron,
        state=response.new_state,
        last_output=response.activation_up,
    )
