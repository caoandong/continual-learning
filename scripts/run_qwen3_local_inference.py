#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CHECKPOINT_DIR = Path("/content/drive/MyDrive/flair/software/qwen3/checkpoints")
DEFAULT_SYSTEM_PROMPT = "You are a concise assistant."


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    expected_substring: str | None = None


DEFAULT_CASES = (
    PromptCase(
        prompt="What is 2 + 2? Reply with the numeral only.",
        expected_substring="4",
    ),
    PromptCase(
        prompt="What is the capital of France? Reply with one word.",
        expected_substring="paris",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Optional prompt to run. Repeat to execute multiple prompts. Defaults to built-in smoke tests.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if device == "mps" and (
        getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but no MPS device is available.")
    return device


def build_cases(prompts: Sequence[str] | None) -> list[PromptCase]:
    if prompts:
        return [PromptCase(prompt=prompt) for prompt in prompts]
    return list(DEFAULT_CASES)


def model_kwargs_for_device(device: str) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": 0}
    elif device == "mps":
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32
    return kwargs


def checkpoint_dir_has_files(checkpoint_dir: Path) -> bool:
    return checkpoint_dir.is_dir() and (checkpoint_dir / "config.json").exists() and any(
        (checkpoint_dir / filename).exists()
        for filename in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    )


def resolve_checkpoint_dir(checkpoint_dir: Path) -> Path:
    if checkpoint_dir_has_files(checkpoint_dir):
        return checkpoint_dir
    child_candidates = sorted(
        child for child in checkpoint_dir.iterdir() if child.is_dir() and checkpoint_dir_has_files(child)
    )
    if len(child_candidates) == 1:
        return child_candidates[0]
    if len(child_candidates) > 1:
        raise RuntimeError(
            "Checkpoint directory is ambiguous. Pass --checkpoint-dir pointing at one model snapshot. "
            f"Candidates: {', '.join(str(path) for path in child_candidates)}"
        )
    return checkpoint_dir


def normalize_device_placement(placement: object) -> str | None:
    if isinstance(placement, int):
        return f"cuda:{placement}"
    if isinstance(placement, str):
        return placement
    if isinstance(placement, torch.device):
        return str(placement)
    return None


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for placement in device_map.values():
            normalized = normalize_device_placement(placement)
            if normalized and normalized != "disk":
                return torch.device(normalized)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_model_device_map(model: AutoModelForCausalLM) -> dict[str, str]:
    raw_map = getattr(model, "hf_device_map", None)
    if isinstance(raw_map, dict):
        normalized: dict[str, str] = {}
        for name, placement in raw_map.items():
            normalized[str(name)] = normalize_device_placement(placement) or str(placement)
        return normalized
    return {"": str(get_input_device(model))}


def build_generation_config(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> object:
    do_sample = temperature > 0.0
    generation_config = copy.deepcopy(model.generation_config)
    eos_token_id = generation_config.eos_token_id or tokenizer.eos_token_id
    pad_token_id = generation_config.pad_token_id or tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    generation_config.eos_token_id = eos_token_id
    generation_config.pad_token_id = pad_token_id
    if do_sample:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    return generation_config


def assert_expected(case: PromptCase, completion: str) -> None:
    if case.expected_substring and case.expected_substring.lower() not in completion.lower():
        raise RuntimeError(
            f"Smoke test failed for prompt={case.prompt!r}. "
            f"Expected substring {case.expected_substring!r}, got {completion!r}."
        )


def load_model_and_tokenizer(
    checkpoint_dir: Path,
    device: str,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs_for_device(device))
    if device != "cuda":
        model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def run_prompt_cases(
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    system_prompt: str,
    cases: Iterable[PromptCase],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    input_device = get_input_device(model)
    generation_config = build_generation_config(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    for case in cases:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt").to(input_device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, generation_config=generation_config)
        elapsed = time.perf_counter() - started
        completion_ids = generated[0][encoded.input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        assert_expected(case, completion)
        results.append(
            {
                "prompt": case.prompt,
                "expected_substring": case.expected_substring,
                "completion": completion,
                "latency_seconds": round(elapsed, 3),
            }
        )
    return results


def collect_runtime_metadata(model: AutoModelForCausalLM, resolved_device: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "resolved_device": resolved_device,
        "model_device_map": get_model_device_map(model),
    }
    if resolved_device == "cuda":
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        metadata["gpu"] = {
            "index": device_index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
        }
    return metadata


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)

    resolved_device = resolve_device(args.device)
    cases = build_cases(args.prompts)
    torch.manual_seed(args.seed)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    load_started = time.perf_counter()
    tokenizer, model = load_model_and_tokenizer(checkpoint_dir=checkpoint_dir, device=resolved_device)
    load_elapsed = time.perf_counter() - load_started
    results = run_prompt_cases(
        tokenizer=tokenizer,
        model=model,
        system_prompt=args.system_prompt,
        cases=cases,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "resolved_device": resolved_device,
        "load_seconds": round(load_elapsed, 3),
        "model_type": getattr(model.config, "model_type", None),
        "torch_dtype": str(next(model.parameters()).dtype),
        "runtime": collect_runtime_metadata(model, resolved_device),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
