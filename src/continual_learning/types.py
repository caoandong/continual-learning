from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from continual_learning.constants import DEFAULT_NEURON_STATE, EMPTY_SIGNAL


@dataclass(frozen=True)
class NeuronState:
    name: str
    state_text: str = DEFAULT_NEURON_STATE
    last_latent: str = EMPTY_SIGNAL


@dataclass(frozen=True)
class NeuronResponse:
    state_text: str
    latent_up: str
    task_up: str
    latent_down: str


@dataclass(frozen=True)
class LayerState:
    neurons: tuple[NeuronState, ...]


@dataclass(frozen=True)
class NetworkState:
    layers: tuple[LayerState, ...]
    latent_activations: tuple[tuple[str, ...], ...]
    task_outputs: tuple[tuple[str, ...], ...]
    latent_feedbacks: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class NetworkStepInput:
    raw_input: str
    top_down_feedback: str = EMPTY_SIGNAL
    teaching_signal: str = EMPTY_SIGNAL
    allow_state_update: bool = True


@dataclass(frozen=True)
class NetworkReadout:
    latent_output: str = EMPTY_SIGNAL
    task_output: str = EMPTY_SIGNAL


@dataclass(frozen=True)
class NetworkStepResult:
    state: NetworkState
    latent_output: str = EMPTY_SIGNAL
    task_output: str = EMPTY_SIGNAL


LlmCaller = Callable[[str], NeuronResponse]
BatchLlmCaller = Callable[[tuple[str, ...]], tuple[NeuronResponse, ...]]
