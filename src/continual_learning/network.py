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

_step_counter = 0


@dataclass(frozen=True)
class LayerStepResult:
    layer: LayerState
    activations: tuple[str, ...]
    feedbacks: tuple[str, ...]


# ---------------------------------------------------------------------------
# State formatting helpers
# ---------------------------------------------------------------------------

def format_neuron_state(neuron: NeuronState) -> str:
    return (
        f"    name={neuron.name}\n"
        f"    state={neuron.state}\n"
        f"    last_output={neuron.last_output}"
    )


def format_network_state(state: NetworkState) -> str:
    lines = []
    for layer_idx, layer in enumerate(state.layers):
        lines.append(f"  Layer {layer_idx} ({len(layer.neurons)} neurons):")
        for neuron in layer.neurons:
            lines.append(format_neuron_state(neuron))
        lines.append(f"    activations={state.activations[layer_idx]}")
        lines.append(f"    feedbacks={state.feedbacks[layer_idx]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer helpers
# ---------------------------------------------------------------------------

def build_layer_bottom_up(
    network_state: NetworkState, layer_index: int, raw_input: str,
) -> str:
    if layer_index == 0:
        result = raw_input
    else:
        result = " | ".join(network_state.activations[layer_index - 1])
    logger.debug(
        "[network] build_layer_bottom_up layer=%d -> %s",
        layer_index, result,
    )
    return result


def build_layer_top_down(
    network_state: NetworkState, layer_index: int, top_down_feedback: str,
) -> str:
    if layer_index == len(network_state.layers) - 1:
        result = top_down_feedback
    else:
        result = " | ".join(network_state.feedbacks[layer_index + 1])
    logger.debug(
        "[network] build_layer_top_down layer=%d -> %s",
        layer_index, result,
    )
    return result


def step_single_neuron(
    neuron: NeuronState,
    step_input: NeuronStepInput,
    call_llm: LlmCaller,
) -> tuple[NeuronState, str, str]:
    prompt = build_neuron_prompt(neuron, step_input)
    logger.debug(
        "[network] step_single_neuron %s\n"
        "  PROMPT:\n%s",
        neuron.name, prompt,
    )
    response = call_llm(prompt)
    logger.debug(
        "[network] step_single_neuron %s\n"
        "  RESPONSE:\n"
        "    new_state=%s\n"
        "    activation_up=%s\n"
        "    feedback_down=%s",
        neuron.name,
        response.new_state,
        response.activation_up,
        response.feedback_down,
    )
    new_neuron = apply_neuron_response(neuron, response)
    logger.debug(
        "[network] step_single_neuron %s\n"
        "  STATE AFTER:\n%s",
        neuron.name, format_neuron_state(new_neuron),
    )
    return new_neuron, response.activation_up, response.feedback_down


def step_layer(
    layer: LayerState,
    bottom_up: str,
    top_down: str,
    call_llm: LlmCaller,
    layer_index: int = -1,
) -> LayerStepResult:
    logger.debug(
        "[network] step_layer %d\n"
        "  bottom_up=%s\n"
        "  top_down=%s\n"
        "  neurons=%s",
        layer_index,
        bottom_up,
        top_down,
        tuple(n.name for n in layer.neurons),
    )
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
    logger.debug(
        "[network] step_layer %d RESULT\n"
        "  activations=%s\n"
        "  feedbacks=%s",
        layer_index, activations, feedbacks,
    )
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
    state = NetworkState(
        layers=tuple(layers),
        activations=tuple(activations),
        feedbacks=tuple(feedbacks),
    )
    logger.debug(
        "[network] create_network_state layer_sizes=%s\n"
        "  INITIAL STATE:\n%s",
        layer_sizes, format_network_state(state),
    )
    return state


def step_network(
    state: NetworkState,
    step_input: NetworkStepInput,
    call_llm: LlmCaller,
) -> NetworkStepResult:
    global _step_counter  # noqa: PLW0603
    _step_counter += 1
    step_num = _step_counter

    logger.debug(
        "\n"
        "============================================================\n"
        "  NETWORK STEP %d\n"
        "============================================================\n"
        "  raw_input=%s\n"
        "  top_down_feedback=%s\n"
        "------------------------------------------------------------\n"
        "  STATE BEFORE:\n%s\n"
        "------------------------------------------------------------",
        step_num,
        step_input.raw_input,
        step_input.top_down_feedback,
        format_network_state(state),
    )

    def update_layer(index: int) -> LayerStepResult:
        bottom_up = build_layer_bottom_up(state, index, step_input.raw_input)
        top_down = build_layer_top_down(state, index, step_input.top_down_feedback)
        return step_layer(state.layers[index], bottom_up, top_down, call_llm, index)

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

    logger.debug(
        "------------------------------------------------------------\n"
        "  NETWORK STEP %d RESULT\n"
        "------------------------------------------------------------\n"
        "  prediction=%s\n"
        "  STATE AFTER:\n%s\n"
        "============================================================",
        step_num,
        prediction,
        format_network_state(new_state),
    )
    return NetworkStepResult(state=new_state, prediction=prediction)
