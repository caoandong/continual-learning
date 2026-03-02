from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from continual_learning.constants import EMPTY_SIGNAL


@dataclass(frozen=True)
class NeuronState:
    name: str
    state: str = ""
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


LlmCaller = Callable[[str], NeuronResponse]
