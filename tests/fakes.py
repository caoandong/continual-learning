from __future__ import annotations

import re

from continual_learning.constants import EMPTY_SIGNAL
from continual_learning.types import NeuronResponse


def extract_prompt_section(prompt: str, section_start: str, section_end: str) -> str:
    pattern = re.escape(section_start) + r"\s*(.*?)\s*" + re.escape(section_end)
    match = re.search(pattern, prompt, re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip()


def extract_prompt_field(prompt: str, field_name: str) -> str:
    pattern = r"^" + re.escape(field_name) + r":[ \t]*(.*)$"
    match = re.search(pattern, prompt, re.MULTILINE)
    if match is None:
        return ""
    return match.group(1).strip()


def parse_mappings(state_text: str) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for line in state_text.splitlines():
        if " => " not in line:
            continue
        key, value = line.split(" => ", 1)
        mappings[key] = value
    return mappings


def serialize_mappings(mappings: dict[str, str]) -> str:
    return "\n".join(f"{key} => {value}" for key, value in sorted(mappings.items()))


def split_segments(text: str) -> list[str]:
    if text in ("", EMPTY_SIGNAL):
        return []
    return [segment.strip() for segment in text.split(" | ") if segment.strip()]


def combine_signal(last_latent: str, bottom_up: str) -> str:
    segments = split_segments(last_latent)
    if bottom_up in ("", EMPTY_SIGNAL):
        return " | ".join(segments)
    if not segments or segments[-1] != bottom_up:
        segments.append(bottom_up)
    return " | ".join(segments)


def simple_learning_response(prompt: str) -> NeuronResponse:
    state_text = extract_prompt_section(prompt, "CURRENT STATE:", "INPUTS:")
    bottom_up = extract_prompt_field(prompt, "Bottom-up context")
    teaching_signal = extract_prompt_field(prompt, "Teaching signal")
    mode = extract_prompt_field(prompt, "State update mode")
    last_latent = extract_prompt_field(prompt, "Last latent output")
    mappings = parse_mappings(state_text)
    latent = combine_signal(last_latent, bottom_up)

    if mode == "write" and teaching_signal not in ("", EMPTY_SIGNAL) and latent:
        mappings[latent] = teaching_signal
        state_text = serialize_mappings(mappings)

    task_output = mappings.get(latent, EMPTY_SIGNAL)
    if mode == "write" and teaching_signal not in ("", EMPTY_SIGNAL):
        task_output = teaching_signal

    return NeuronResponse(
        state_text=state_text,
        latent_up=latent,
        task_up=task_output,
        latent_down=EMPTY_SIGNAL,
    )


def simple_learning_batch(prompts: tuple[str, ...]) -> tuple[NeuronResponse, ...]:
    return tuple(simple_learning_response(prompt) for prompt in prompts)
