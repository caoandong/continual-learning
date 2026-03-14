from __future__ import annotations

from types import SimpleNamespace

import litellm

from continual_learning.llm import (
    StructuredNeuronResponseModel,
    call_llm_batch_threaded,
    call_llm_litellm,
    parse_structured_message,
)
from continual_learning.types import NeuronResponse


def test_parse_structured_message_accepts_parsed_model() -> None:
    expected = StructuredNeuronResponseModel(
        state_text="next",
        latent_up="up",
        task_up="task",
        latent_down="down",
    )
    message = SimpleNamespace(parsed=expected)
    assert parse_structured_message(message) == expected


def test_parse_structured_message_accepts_dict_content() -> None:
    message = SimpleNamespace(
        parsed=None,
        content={
            "state_text": "next",
            "latent_up": "up",
            "task_up": "task",
            "latent_down": "down",
        },
    )
    parsed = parse_structured_message(message)
    assert parsed.state_text == "next"
    assert parsed.latent_up == "up"
    assert parsed.task_up == "task"
    assert parsed.latent_down == "down"


def test_call_llm_litellm_returns_direct_structured_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_completion(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content={
                            "state_text": "next state",
                            "latent_up": "signal",
                            "task_up": "task",
                            "latent_down": "feedback",
                        },
                    ),
                ),
            ],
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    response = call_llm_litellm("prompt text", model="test-model")
    assert response == NeuronResponse(
        state_text="next state",
        latent_up="signal",
        task_up="task",
        latent_down="feedback",
    )
    assert captured["model"] == "test-model"
    assert captured["response_format"] is StructuredNeuronResponseModel


def test_call_llm_batch_threaded_preserves_prompt_order() -> None:
    def echo(prompt: str) -> NeuronResponse:
        return NeuronResponse(
            state_text=prompt,
            latent_up=prompt,
            task_up=prompt,
            latent_down=prompt,
        )

    prompts = ("one", "two", "three")
    responses = call_llm_batch_threaded(prompts, echo)
    assert tuple(response.latent_up for response in responses) == prompts
