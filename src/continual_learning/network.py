from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass

from continual_learning.constants import EMPTY_SIGNAL
from continual_learning.neuron import apply_neuron_response, build_neuron_prompt
from continual_learning.types import (
    LayerState,
    LlmCaller,
    NetworkState,
    NetworkStepInput,
    NetworkStepResult,
    NeuronState,
    NeuronStepInput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerStepResult:
    layer: LayerState
    activations: tuple[str, ...]
    feedbacks: tuple[str, ...]


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------

def build_layer_bottom_up(
    network_state: NetworkState, layer_index: int, raw_input: str,
) -> str:
    if layer_index == 0:
        return raw_input
    return " | ".join(network_state.activations[layer_index - 1])


def build_layer_top_down(
    network_state: NetworkState, layer_index: int, top_down_feedback: str,
) -> str:
    if layer_index == len(network_state.layers) - 1:
        return top_down_feedback
    return " | ".join(network_state.feedbacks[layer_index + 1])


def step_single_neuron(
    neuron: NeuronState,
    step_input: NeuronStepInput,
    call_llm: LlmCaller,
) -> tuple[NeuronState, str, str]:
    prompt = build_neuron_prompt(neuron, step_input)
    response = call_llm(prompt)
    new_neuron = apply_neuron_response(neuron, response)
    return new_neuron, response.activation_up, response.feedback_down


def step_layer(
    layer: LayerState,
    bottom_up: str,
    top_down: str,
    call_llm: LlmCaller,
) -> LayerStepResult:
    step_input = NeuronStepInput(bottom_up=bottom_up, top_down=top_down)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(layer.neurons)) as executor:
        futures = [
            executor.submit(step_single_neuron, neuron, step_input, call_llm)
            for neuron in layer.neurons
        ]
        results = [f.result() for f in futures]

    new_neurons = tuple(r[0] for r in results)
    activations = tuple(r[1] for r in results)
    feedbacks = tuple(r[2] for r in results)
    return LayerStepResult(
        layer=LayerState(neurons=new_neurons),
        activations=activations,
        feedbacks=feedbacks,
    )


# ---------------------------------------------------------------------------
# Network operations
# ---------------------------------------------------------------------------

def create_network_state(layer_sizes: tuple[int, ...]) -> NetworkState:
    layers: list[LayerState] = []
    activations: list[tuple[str, ...]] = []
    feedbacks: list[tuple[str, ...]] = []
    for layer_index, size in enumerate(layer_sizes):
        neurons = tuple(
            NeuronState(name=f"L{layer_index}_N{i}") for i in range(size)
        )
        layers.append(LayerState(neurons=neurons))
        activations.append(tuple(EMPTY_SIGNAL for _ in range(size)))
        feedbacks.append(tuple(EMPTY_SIGNAL for _ in range(size)))
    return NetworkState(
        layers=tuple(layers),
        activations=tuple(activations),
        feedbacks=tuple(feedbacks),
    )


def step_network(
    state: NetworkState,
    step_input: NetworkStepInput,
    call_llm: LlmCaller,
) -> NetworkStepResult:
    def update_layer(index: int) -> LayerStepResult:
        bottom_up = build_layer_bottom_up(state, index, step_input.raw_input)
        top_down = build_layer_top_down(state, index, step_input.top_down_feedback)
        return step_layer(state.layers[index], bottom_up, top_down, call_llm)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(state.layers)) as executor:
        futures = {
            index: executor.submit(update_layer, index)
            for index in range(len(state.layers))
        }
        layer_results = {index: f.result() for index, f in futures.items()}

    new_layers = tuple(layer_results[i].layer for i in range(len(state.layers)))
    new_activations = tuple(layer_results[i].activations for i in range(len(state.layers)))
    new_feedbacks = tuple(layer_results[i].feedbacks for i in range(len(state.layers)))

    new_state = NetworkState(
        layers=new_layers,
        activations=new_activations,
        feedbacks=new_feedbacks,
    )
    prediction = new_activations[-1][0] if new_activations[-1] else "unknown"

    logger.debug("[network] step prediction=%s", prediction)
    return NetworkStepResult(state=new_state, prediction=prediction)
