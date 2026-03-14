from __future__ import annotations

import functools
import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ConfigDict

from continual_learning.constants import DEFAULT_TEMPERATURE, NEURON_SYSTEM_PROMPT
from continual_learning.environment import load_environment_file
from continual_learning.types import BatchLlmCaller, LlmCaller, NeuronResponse

logger = logging.getLogger(__name__)


class StructuredNeuronResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_text: str
    latent_up: str
    task_up: str
    latent_down: str


def parse_structured_message(message: object) -> StructuredNeuronResponseModel:
    parsed = getattr(message, "parsed", None)
    if isinstance(parsed, StructuredNeuronResponseModel):
        return parsed

    content = getattr(message, "content", None)
    if isinstance(content, StructuredNeuronResponseModel):
        return content
    if isinstance(content, str):
        return StructuredNeuronResponseModel.model_validate_json(content)
    if isinstance(content, dict):
        return StructuredNeuronResponseModel.model_validate(content)
    raise TypeError(f"Unsupported LiteLLM message payload: {type(content)!r}")


def call_llm_litellm(prompt: str, *, model: str) -> NeuronResponse:
    import litellm  # noqa: PLC0415

    logger.debug("[llm] call_llm_litellm model=%s FULL PROMPT:\n%s", model, prompt)
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": NEURON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=DEFAULT_TEMPERATURE,
        response_format=StructuredNeuronResponseModel,
    )
    message = response.choices[0].message  # type: ignore[union-attr]
    parsed = parse_structured_message(message)
    logger.debug("[llm] call_llm_litellm PARSED RESPONSE:\n%s", parsed.model_dump_json())
    return NeuronResponse(
        state_text=parsed.state_text,
        latent_up=parsed.latent_up,
        task_up=parsed.task_up,
        latent_down=parsed.latent_down,
    )


def create_llm_caller(*, model: str) -> LlmCaller:
    load_environment_file()
    logger.info("[llm] Using litellm with model=%s", model)
    return functools.partial(call_llm_litellm, model=model)


def call_llm_batch_threaded(
    prompts: tuple[str, ...],
    call_llm: LlmCaller,
) -> tuple[NeuronResponse, ...]:
    if not prompts:
        return ()
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(call_llm, prompt) for prompt in prompts]
        return tuple(future.result() for future in futures)


def create_batch_llm_caller(*, model: str) -> BatchLlmCaller:
    call_llm = create_llm_caller(model=model)
    return functools.partial(call_llm_batch_threaded, call_llm=call_llm)
