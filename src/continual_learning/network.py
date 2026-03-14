from __future__ import annotations

import logging
import random
from dataclasses import replace

from continual_learning.constants import EMPTY_SIGNAL, RANDOM_STATE_SEED
from continual_learning.neuron import apply_neuron_response, build_neuron_prompt
from continual_learning.state import build_random_state_text, render_signal
from continual_learning.types import (
    BatchLlmCaller,
    LayerState,
    NetworkReadout,
    NetworkState,
    NetworkStepInput,
    NetworkStepResult,
    NeuronResponse,
    NeuronState,
)

logger = logging.getLogger(__name__)


def format_neuron_state(neuron: NeuronState) -> str:
    return (
        f"    name={neuron.name}\n"
        f"    state_text={render_signal(neuron.state_text)}\n"
        f"    last_latent={render_signal(neuron.last_latent)}"
    )


def format_network_state(state: NetworkState) -> str:
    lines: list[str] = []
    for layer_index, layer in enumerate(state.layers):
        lines.append(f"  Layer {layer_index}:")
        for neuron in layer.neurons:
            lines.append(format_neuron_state(neuron))
        lines.append(f"    latent_activations={state.latent_activations[layer_index]}")
        lines.append(f"    task_outputs={state.task_outputs[layer_index]}")
        lines.append(f"    latent_feedbacks={state.latent_feedbacks[layer_index]}")
    return "\n".join(lines)


def join_signals(signals: tuple[str, ...]) -> str:
    active_signals = tuple(signal for signal in signals if signal)
    if active_signals:
        return " | ".join(active_signals)
    return EMPTY_SIGNAL


def build_layer_bottom_up(
    network_state: NetworkState,
    layer_index: int,
    raw_input: str,
) -> str:
    if layer_index == 0:
        return raw_input
    return join_signals(network_state.latent_activations[layer_index - 1])


def build_layer_top_down(
    network_state: NetworkState,
    layer_index: int,
    top_down_feedback: str,
) -> str:
    if layer_index == len(network_state.layers) - 1:
        return top_down_feedback
    return join_signals(network_state.latent_feedbacks[layer_index + 1])


def collect_prompts(
    state: NetworkState,
    step_input: NetworkStepInput,
) -> tuple[tuple[int, int, NeuronState, str], ...]:
    collected: list[tuple[int, int, NeuronState, str]] = []
    for layer_index, layer in enumerate(state.layers):
        bottom_up = build_layer_bottom_up(state, layer_index, step_input.raw_input)
        top_down = build_layer_top_down(state, layer_index, step_input.top_down_feedback)
        teaching_signal = (
            step_input.teaching_signal
            if layer_index == len(state.layers) - 1
            else EMPTY_SIGNAL
        )
        for neuron_index, neuron in enumerate(layer.neurons):
            prompt = build_neuron_prompt(
                neuron,
                bottom_up=bottom_up,
                top_down=top_down,
                teaching_signal=teaching_signal,
                allow_state_update=step_input.allow_state_update,
            )
            collected.append((layer_index, neuron_index, neuron, prompt))
    return tuple(collected)


def apply_all_responses(
    state: NetworkState,
    collected: tuple[tuple[int, int, NeuronState, str], ...],
    responses: tuple[NeuronResponse, ...],
) -> NetworkStepResult:
    if len(collected) != len(responses):
        raise ValueError("LLM batch size did not match the number of neuron prompts")

    grouped: list[list[tuple[int, NeuronState, NeuronResponse]]] = [[] for _ in state.layers]
    for (layer_index, neuron_index, neuron, _), response in zip(collected, responses, strict=True):
        grouped[layer_index].append((neuron_index, neuron, response))

    layers: list[LayerState] = []
    latent_activations: list[tuple[str, ...]] = []
    task_outputs: list[tuple[str, ...]] = []
    latent_feedbacks: list[tuple[str, ...]] = []
    for layer_results in grouped:
        ordered = tuple(sorted(layer_results, key=lambda item: item[0]))
        layers.append(
            LayerState(
                neurons=tuple(
                    apply_neuron_response(neuron, response)
                    for _, neuron, response in ordered
                ),
            ),
        )
        latent_activations.append(tuple(response.latent_up for _, _, response in ordered))
        task_outputs.append(tuple(response.task_up for _, _, response in ordered))
        latent_feedbacks.append(tuple(response.latent_down for _, _, response in ordered))

    next_state = NetworkState(
        layers=tuple(layers),
        latent_activations=tuple(latent_activations),
        task_outputs=tuple(task_outputs),
        latent_feedbacks=tuple(latent_feedbacks),
    )
    readout = NetworkReadout(
        latent_output=next_state.latent_activations[-1][0] if next_state.latent_activations[-1] else EMPTY_SIGNAL,
        task_output=next_state.task_outputs[-1][0] if next_state.task_outputs[-1] else EMPTY_SIGNAL,
    )
    return NetworkStepResult(
        state=next_state,
        latent_output=readout.latent_output,
        task_output=readout.task_output,
    )


def build_random_neuron_state(*, name: str, generator: random.Random) -> NeuronState:
    return NeuronState(name=name, state_text=build_random_state_text(generator=generator))


def create_network_state(layer_sizes: tuple[int, ...]) -> NetworkState:
    generator = random.Random(RANDOM_STATE_SEED)
    layers: list[LayerState] = []
    activations: list[tuple[str, ...]] = []
    task_outputs: list[tuple[str, ...]] = []
    feedbacks: list[tuple[str, ...]] = []
    for layer_index, size in enumerate(layer_sizes):
        neurons = tuple(
            build_random_neuron_state(
                name=f"L{layer_index}_N{neuron_index}",
                generator=generator,
            )
            for neuron_index in range(size)
        )
        layers.append(LayerState(neurons=neurons))
        activations.append(tuple(EMPTY_SIGNAL for _ in range(size)))
        task_outputs.append(tuple(EMPTY_SIGNAL for _ in range(size)))
        feedbacks.append(tuple(EMPTY_SIGNAL for _ in range(size)))
    state = NetworkState(
        layers=tuple(layers),
        latent_activations=tuple(activations),
        task_outputs=tuple(task_outputs),
        latent_feedbacks=tuple(feedbacks),
    )
    logger.debug("[network] create_network_state\n%s", format_network_state(state))
    return state


def reset_network_traces(state: NetworkState) -> NetworkState:
    layers = tuple(
        LayerState(
            neurons=tuple(replace(neuron, last_latent=EMPTY_SIGNAL) for neuron in layer.neurons),
        )
        for layer in state.layers
    )
    activations = tuple(tuple(EMPTY_SIGNAL for _ in layer.neurons) for layer in layers)
    task_outputs = tuple(tuple(EMPTY_SIGNAL for _ in layer.neurons) for layer in layers)
    feedbacks = tuple(tuple(EMPTY_SIGNAL for _ in layer.neurons) for layer in layers)
    reset_state = NetworkState(
        layers=layers,
        latent_activations=activations,
        task_outputs=task_outputs,
        latent_feedbacks=feedbacks,
    )
    logger.debug("[network] reset_network_traces\n%s", format_network_state(reset_state))
    return reset_state


def step_network(
    state: NetworkState,
    step_input: NetworkStepInput,
    call_llm_batch: BatchLlmCaller,
) -> NetworkStepResult:
    logger.debug(
        "[network] step_network START input=%s top_down=%s teaching_signal=%s allow_state_update=%s",
        render_signal(step_input.raw_input),
        render_signal(step_input.top_down_feedback),
        render_signal(step_input.teaching_signal),
        step_input.allow_state_update,
    )
    collected = collect_prompts(state, step_input)
    prompts = tuple(prompt for _, _, _, prompt in collected)
    responses = call_llm_batch(prompts)
    result = apply_all_responses(state, collected, responses)
    logger.debug(
        "[network] step_network END latent_output=%s task_output=%s",
        render_signal(result.latent_output),
        render_signal(result.task_output),
    )
    return result
