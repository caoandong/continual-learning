from __future__ import annotations

import logging
import random
from dataclasses import dataclass, replace

from continual_learning.constants import (
    EMPTY_SIGNAL,
    RANDOM_STATE_SEED,
)
from continual_learning.neuron import apply_neuron_response, build_neuron_prompt
from continual_learning.state import build_random_state_text
from continual_learning.types import (
    BatchLlmCaller,
    LayerState,
    NetworkState,
    NetworkStepInput,
    NetworkStepResult,
    NeuronCallRequest,
    NeuronCallResult,
    NeuronState,
    NeuronStepInput,
)

logger = logging.getLogger(__name__)

_step_counter = 0


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


# ---------------------------------------------------------------------------
# Phase 1: Collect all LLM call requests (pure, sync)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerCollectInput:
    layer_index: int
    neurons: tuple[NeuronState, ...]
    step_input: NeuronStepInput


def collect_requests_for_layer(
    layer_input: LayerCollectInput,
) -> tuple[NeuronCallRequest, ...]:
    return tuple(
        NeuronCallRequest(
            layer_index=layer_input.layer_index,
            neuron_index=i,
            neuron=neuron,
            prompt=build_neuron_prompt(neuron, layer_input.step_input),
        )
        for i, neuron in enumerate(layer_input.neurons)
    )


def collect_all_requests(
    state: NetworkState, step_input: NetworkStepInput,
) -> tuple[NeuronCallRequest, ...]:
    all_requests: list[NeuronCallRequest] = []
    for layer_index in range(len(state.layers)):
        layer_input = LayerCollectInput(
            layer_index=layer_index,
            neurons=state.layers[layer_index].neurons,
            step_input=NeuronStepInput(
                bottom_up=build_layer_bottom_up(
                    state, layer_index, step_input.raw_input,
                ),
                top_down=build_layer_top_down(
                    state, layer_index, step_input.top_down_feedback,
                ),
                allow_state_update=step_input.allow_state_update,
            ),
        )
        all_requests.extend(collect_requests_for_layer(layer_input))
    return tuple(all_requests)


# ---------------------------------------------------------------------------
# Phase 3: Apply all results (pure, sync)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerApplyResult:
    layer: LayerState
    activations: tuple[str, ...]
    feedbacks: tuple[str, ...]


def apply_results_for_layer(
    results: tuple[NeuronCallResult, ...],
) -> LayerApplyResult:
    new_neurons = tuple(
        apply_neuron_response(r.request.neuron, r.response) for r in results
    )
    activations = tuple(r.response.activation_up for r in results)
    feedbacks = tuple(r.response.feedback_down for r in results)
    return LayerApplyResult(
        layer=LayerState(neurons=new_neurons),
        activations=activations,
        feedbacks=feedbacks,
    )


def group_results_by_layer(
    results: tuple[NeuronCallResult, ...], layer_count: int,
) -> tuple[tuple[NeuronCallResult, ...], ...]:
    grouped: list[list[NeuronCallResult]] = [[] for _ in range(layer_count)]
    for result in results:
        grouped[result.request.layer_index].append(result)
    return tuple(
        tuple(sorted(layer, key=lambda r: r.request.neuron_index))
        for layer in grouped
    )


def apply_all_results(
    state: NetworkState, results: tuple[NeuronCallResult, ...],
) -> NetworkStepResult:
    grouped = group_results_by_layer(results, len(state.layers))
    layer_outputs = tuple(apply_results_for_layer(g) for g in grouped)
    new_state = NetworkState(
        layers=tuple(o.layer for o in layer_outputs),
        activations=tuple(o.activations for o in layer_outputs),
        feedbacks=tuple(o.feedbacks for o in layer_outputs),
    )
    prediction = (
        new_state.activations[-1][0]
        if new_state.activations[-1]
        else "unknown"
    )
    return NetworkStepResult(state=new_state, prediction=prediction)


def build_random_neuron_state(
    *, name: str, generator: random.Random,
) -> NeuronState:
    return NeuronState(name=name, state=build_random_state_text(generator=generator))


# ---------------------------------------------------------------------------
# Network operations
# ---------------------------------------------------------------------------

def create_network_state(layer_sizes: tuple[int, ...]) -> NetworkState:
    generator = random.Random(RANDOM_STATE_SEED)
    layers: list[LayerState] = []
    activations: list[tuple[str, ...]] = []
    feedbacks: list[tuple[str, ...]] = []
    for layer_index, size in enumerate(layer_sizes):
        neurons = tuple(
            build_random_neuron_state(
                name=f"L{layer_index}_N{i}",
                generator=generator,
            )
            for i in range(size)
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


def reset_network_traces(state: NetworkState) -> NetworkState:
    layers = tuple(
        LayerState(
            neurons=tuple(replace(neuron, last_output=EMPTY_SIGNAL) for neuron in layer.neurons),
        )
        for layer in state.layers
    )
    activations = tuple(
        tuple(EMPTY_SIGNAL for _ in layer.neurons) for layer in layers
    )
    feedbacks = tuple(
        tuple(EMPTY_SIGNAL for _ in layer.neurons) for layer in layers
    )
    reset_state = NetworkState(
        layers=layers,
        activations=activations,
        feedbacks=feedbacks,
    )
    logger.debug(
        "[network] reset_network_traces\n"
        "  RESET STATE:\n%s",
        format_network_state(reset_state),
    )
    return reset_state


def step_network(
    state: NetworkState,
    step_input: NetworkStepInput,
    call_llm_batch: BatchLlmCaller,
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

    # Phase 1: Collect all prompts (pure)
    requests = collect_all_requests(state, step_input)

    # Phase 2: Batch execute all LLM calls (one centralized call)
    prompts = tuple(r.prompt for r in requests)
    responses = call_llm_batch(prompts)
    results = tuple(
        NeuronCallResult(request=req, response=resp)
        for req, resp in zip(requests, responses, strict=True)
    )

    # Phase 3: Apply all responses (pure)
    step_result = apply_all_results(state, results)

    logger.debug(
        "------------------------------------------------------------\n"
        "  NETWORK STEP %d RESULT\n"
        "------------------------------------------------------------\n"
        "  prediction=%s\n"
        "  STATE AFTER:\n%s\n"
        "============================================================",
        step_num,
        step_result.prediction,
        format_network_state(step_result.state),
    )
    return step_result
