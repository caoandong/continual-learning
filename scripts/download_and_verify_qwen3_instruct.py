#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CHECKPOINT_DIR = Path("/content/drive/MyDrive/flair/software/qwen3/checkpoints")

SMOKE_TESTS: List[Dict[str, str]] = [
    {
        "prompt": "What is 2 + 2? Reply with the numeral only.",
        "expected": "4",
    },
    {
        "prompt": "What is the capital of France? Reply with one word.",
        "expected": "paris",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def model_kwargs_for_device(device: str) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    elif device == "mps":
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32
    return kwargs


def get_model_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def download_snapshot(model_id: str, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=checkpoint_dir,
    )
    return checkpoint_dir


def run_smoke_tests(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    generation_config = model.generation_config
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    for case in SMOKE_TESTS:
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": case["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_device = get_model_device(model) if device == "cuda" else torch.device(device)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        if case["expected"].lower() not in completion.lower():
            raise RuntimeError(
                f"Smoke test failed for prompt={case['prompt']!r}. "
                f"Expected substring {case['expected']!r}, got {completion!r}."
            )
        results.append(
            {
                "prompt": case["prompt"],
                "expected": case["expected"],
                "completion": completion,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not args.skip_download:
        download_snapshot(args.model_id, checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, **model_kwargs_for_device(device))
    if device != "cuda":
        model.to(torch.device(device))
    model.eval()

    results = run_smoke_tests(
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "checkpoint_dir": str(checkpoint_dir),
                "device": device,
                "tests": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
