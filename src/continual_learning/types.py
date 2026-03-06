from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from continual_learning.constants import DEFAULT_NEURON_STATE, EMPTY_SIGNAL


@dataclass(frozen=True)
class NeuronState:
    name: str
    state: str = DEFAULT_NEURON_STATE
    last_output: str = EMPTY_SIGNAL


@dataclass(frozen=True)
class NeuronResponse:
    new_state: str
    activation_up: str
    feedback_down: str


@dataclass(frozen=True)
class NeuronStepInput:
    bottom_up: str
    top_down: str
    sensory_input: str = ""
    allow_state_update: bool = True


@dataclass(frozen=True)
class LayerState:
    neurons: tuple[NeuronState, ...]


@dataclass(frozen=True)
class NetworkState:
    layers: tuple[LayerState, ...]
    activations: tuple[tuple[str, ...], ...]
    feedbacks: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class NetworkStepInput:
    raw_input: str
    top_down_feedback: str
    allow_state_update: bool = True


@dataclass(frozen=True)
class NetworkStepResult:
    state: NetworkState
    prediction: str


@dataclass(frozen=True)
class ExperimentOptions:
    model: str
    use_mock: bool
    layer_sizes: tuple[int, ...]
    verbose: bool


@dataclass(frozen=True)
class NeuronCallRequest:
    layer_index: int
    neuron_index: int
    neuron: NeuronState
    prompt: str


@dataclass(frozen=True)
class NeuronCallResult:
    request: NeuronCallRequest
    response: NeuronResponse


LlmCaller = Callable[[str], NeuronResponse]
BatchLlmCaller = Callable[[tuple[str, ...]], tuple[NeuronResponse, ...]]
