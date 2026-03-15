#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch

from nanoqwen.model import (
    QWEN3_CONFIGS,
    Qwen3Model,
    build_empty_thought_patches,
    generate_text_simple,
    load_model_and_tokenizer,
)


LOGGER = logging.getLogger("qwen_thought_patch")
DEFAULT_LOCAL_CHECKPOINTS = {
    "0.6B": Path("/Volumes/SB-XTM5/flair/software/qwen3/checkpoints/Qwen3-0.6B"),
}


@dataclass(frozen=True)
class ArithmeticExample:
    numbers: tuple[int, int, int]
    answer: int

    @property
    def raw_query(self) -> str:
        a, b, c = self.numbers
        return f"Query: {a}, {b}, {c}."


@dataclass(frozen=True)
class TaskSpec:
    slug: str
    title: str
    instruction: str

    def contextual_prompt(self, example: ArithmeticExample) -> str:
        return f"{self.instruction} {example.raw_query}"


TASKS: Dict[str, TaskSpec] = {
    "multiply": TaskSpec(
        slug="multiply",
        title="Multiply numbers",
        instruction="Instruction: multiply the numbers. Answer with one integer only.",
    ),
    "sum": TaskSpec(
        slug="sum",
        title="Sum numbers",
        instruction="Instruction: sum the numbers. Answer with one integer only.",
    ),
}


def reset_patches(patches) -> None:
    for patch in patches:
        patch.zero_()


def patch_norms(patches) -> Dict[str, float]:
    norms = {"fc1": 0.0, "fc2": 0.0, "fc3": 0.0, "bias": 0.0}
    for patch in patches:
        patch_values = patch.norms()
        for key in norms:
            norms[key] += patch_values[key]
    return norms


@torch.no_grad()
def capture_traces(model: Qwen3Model, input_ids: torch.Tensor, *, device: str, thought_patches=None):
    _, traces = model(input_ids.to(device), thought_patches=thought_patches, return_trace=True)
    return [{key: value.detach().float().cpu() for key, value in trace.items()} for trace in traces]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thought-patching benchmark for Qwen on arithmetic instruction tasks.")
    parser.add_argument("--model-size", choices=sorted(QWEN3_CONFIGS), default="0.6B")
    parser.add_argument("--model-type", choices=["base", "instruct", "reasoning"], default="instruct")
    parser.add_argument("--repo-id", default=None, help="Optional Hugging Face repo id.")
    parser.add_argument("--local-dir", default=None, help="Optional local cache directory for weights/tokenizer.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    parser.add_argument("--tasks", default="multiply,sum", help="Comma-separated task list: multiply,sum")
    parser.add_argument("--train-examples", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=20)
    parser.add_argument("--digit-min", type=int, default=1)
    parser.add_argument("--digit-max", type=int, default=9)
    parser.add_argument("--paper-filter", action="store_true", default=True, help="Filter candidates so prompted model is correct and raw model is incorrect before sampling train/eval splits.")
    parser.add_argument("--no-paper-filter", action="store_false", dest="paper_filter")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--rho", type=float, default=0.0, help="Ridge penalty. Use 0.0 for plain least-squares.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Patch scale multiplier applied to each learned update.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fit-mode", choices=["sequential", "layer_batch"], default="layer_batch")
    parser.add_argument("--patch-fc3", action="store_true", default=False, help="Also solve a down-projection update. Disabled by default because the exact skip-connection theorem only needs first-layer matrices plus a bias-like shift.")
    parser.set_defaults(eval_every_step=True)
    parser.add_argument("--no-step-eval", action="store_false", dest="eval_every_step")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--out", type=Path, default=Path("runs/qwen_thought_patch_metrics.json"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def count_parameters(model: Qwen3Model) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def build_model(
    *,
    model_size: str,
    model_type: str,
    repo_id: str | None,
    local_dir: str | None,
    device: str,
    dtype: torch.dtype,
) -> tuple[Qwen3Model, Any, str, str]:
    is_base = model_type == "base"
    if repo_id is None:
        repo_id = f"Qwen/Qwen3-{model_size}-Base" if is_base else f"Qwen/Qwen3-{model_size}"
    if local_dir is None:
        default_local_dir = DEFAULT_LOCAL_CHECKPOINTS.get(model_size)
        local_dir = str(default_local_dir) if default_local_dir is not None and default_local_dir.exists() else Path(repo_id).parts[-1]
    LOGGER.info("Loading model: repo=%s local_dir=%s device=%s dtype=%s", repo_id, local_dir, device, str(dtype).replace("torch.", ""))
    start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        model_size=model_size,
        model_type=model_type,
        device=device,
        repo_id=repo_id,
        local_dir=local_dir,
        dtype=dtype,
    )
    elapsed = time.perf_counter() - start
    LOGGER.info("Loaded model parameters: %.2fM params in %.1fs", count_parameters(model) / 1_000_000, elapsed)
    return model, tokenizer, repo_id, local_dir


def solve_weight_update(src: torch.Tensor, target: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Solve min_D ||src @ D.T - target||_F^2 + rho ||D||_F^2.
    src: [n, d_in]
    target: [n, d_out]
    returns: [d_out, d_in]
    """
    src32 = src.float()
    tgt32 = target.float()
    if src32.ndim != 2 or tgt32.ndim != 2:
        raise ValueError("solve_weight_update expects rank-2 tensors.")
    if src32.shape[0] != tgt32.shape[0]:
        raise ValueError(f"Mismatched batch dimension: src={src32.shape} target={tgt32.shape}")

    if rho > 0:
        gram = src32.T @ src32
        eye = torch.eye(gram.size(0), dtype=gram.dtype)
        rhs = src32.T @ tgt32
        solution_t = torch.linalg.solve(gram + rho * eye, rhs)
        return solution_t.T.contiguous()

    solution = torch.linalg.lstsq(src32, tgt32).solution
    return solution.T.contiguous()


def sample_examples(task_slug: str, *, train_examples: int, eval_examples: int, seed: int, digit_min: int, digit_max: int) -> Dict[str, List[ArithmeticExample]]:
    if task_slug not in TASKS:
        raise KeyError(f"Unknown task: {task_slug}")
    universe = list(itertools.product(range(digit_min, digit_max + 1), repeat=3))
    needed = train_examples + eval_examples
    if needed > len(universe):
        raise ValueError(f"Requested {needed} examples but only {len(universe)} unique triples are available.")
    rng = random.Random(seed)
    rng.shuffle(universe)
    selected = universe[:needed]

    def compute_answer(nums: tuple[int, int, int]) -> int:
        if task_slug == "multiply":
            return math.prod(nums)
        if task_slug == "sum":
            return sum(nums)
        raise KeyError(task_slug)

    items = [ArithmeticExample(numbers=nums, answer=compute_answer(nums)) for nums in selected]
    return {
        "train": items[:train_examples],
        "eval": items[train_examples:],
    }


def make_examples(
    task_slug: str,
    *,
    model: Qwen3Model,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    train_examples: int,
    eval_examples: int,
    seed: int,
    digit_min: int,
    digit_max: int,
    paper_filter: bool,
) -> Dict[str, List[ArithmeticExample]]:
    if not paper_filter:
        return sample_examples(
            task_slug,
            train_examples=train_examples,
            eval_examples=eval_examples,
            seed=seed,
            digit_min=digit_min,
            digit_max=digit_max,
        )

    task = TASKS[task_slug]
    needed = train_examples + eval_examples
    universe = list(itertools.product(range(digit_min, digit_max + 1), repeat=3))
    rng = random.Random(seed)
    rng.shuffle(universe)
    kept: List[ArithmeticExample] = []

    def compute_answer(nums: tuple[int, int, int]) -> int:
        if task_slug == "multiply":
            return math.prod(nums)
        if task_slug == "sum":
            return sum(nums)
        raise KeyError(task_slug)

    LOGGER.info("[%s seed=%d] selecting %d filtered examples with prompted-correct/raw-incorrect constraint", task_slug, seed, needed)
    for nums in universe:
        example = ArithmeticExample(numbers=nums, answer=compute_answer(nums))
        prompted = evaluate_examples(
            model,
            tokenizer,
            None,
            [example],
            task,
            device=device,
            max_new_tokens=max_new_tokens,
            mode="prompted",
        )["rows"][0]
        raw = evaluate_examples(
            model,
            tokenizer,
            None,
            [example],
            task,
            device=device,
            max_new_tokens=max_new_tokens,
            mode="raw",
        )["rows"][0]
        if prompted["correct"] and not raw["correct"]:
            kept.append(example)
            if len(kept) >= needed:
                break

    if len(kept) < needed:
        raise RuntimeError(
            f"Unable to find {needed} filtered examples for task={task_slug}. Found only {len(kept)}."
        )

    return {
        "train": kept[:train_examples],
        "eval": kept[train_examples:],
    }


def build_teacher_forcing_example(tokenizer: Any, user_text: str, answer_text: str) -> tuple[torch.Tensor, List[int]]:
    prefix_ids = tokenizer.encode(user_text, chat_wrapped=True)
    answer_ids = tokenizer.encode(answer_text, chat_wrapped=False)
    if len(answer_ids) == 0:
        raise ValueError("Answer tokenization produced zero ids.")
    full_ids = prefix_ids + answer_ids + [tokenizer.eos_token_id]
    answer_positions = list(range(len(prefix_ids), len(prefix_ids) + len(answer_ids)))
    return torch.tensor(full_ids, dtype=torch.long).unsqueeze(0), answer_positions


def find_subsequence_positions(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle:
        return []
    needle_len = len(needle)
    for start in range(len(haystack) - needle_len + 1):
        if list(haystack[start : start + needle_len]) == list(needle):
            return list(range(start, start + needle_len))
    raise ValueError("Failed to find token subsequence for alignment.")


def build_alignment_positions(tokenizer: Any, user_text: str, query_text: str, answer_text: str) -> tuple[torch.Tensor, List[int]]:
    full_ids, answer_positions = build_teacher_forcing_example(tokenizer, user_text, answer_text)
    prompt_ids = tokenizer.encode(user_text, chat_wrapped=True)
    query_positions = None
    for variant in (query_text, f" {query_text}", f"\n{query_text}"):
        query_ids = tokenizer.encode(variant, chat_wrapped=False)
        try:
            query_positions = find_subsequence_positions(prompt_ids, query_ids)
            break
        except ValueError:
            continue
    if query_positions is None:
        raise ValueError("Failed to find aligned query token positions in prompt.")
    return full_ids, query_positions + answer_positions


def build_generation_prompt(tokenizer: Any, user_text: str) -> torch.Tensor:
    ids = tokenizer.encode(user_text, chat_wrapped=True)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def decode_generated_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(list(token_ids)).strip()


def normalize_prediction(text: str) -> str:
    integers = re.findall(r"-?\d+", text)
    if integers:
        return integers[-1]
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def generate_answer(
    model: Qwen3Model,
    tokenizer: Any,
    *,
    user_text: str,
    device: str,
    max_new_tokens: int,
    thought_patches=None,
) -> Dict[str, Any]:
    prompt_ids = build_generation_prompt(tokenizer, user_text).to(device)
    generated: List[int] = []
    start = time.perf_counter()
    for next_token in generate_text_simple(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        thought_patches=thought_patches,
    ):
        generated.append(int(next_token.item()))
    elapsed = time.perf_counter() - start
    decoded = decode_generated_text(tokenizer, generated)
    return {
        "raw_text": decoded,
        "normalized": normalize_prediction(decoded),
        "new_tokens": len(generated),
        "elapsed_s": elapsed,
    }


def evaluate_examples(
    model: Qwen3Model,
    tokenizer: Any,
    thought_patches,
    examples: Sequence[ArithmeticExample],
    task: TaskSpec,
    *,
    device: str,
    max_new_tokens: int,
    mode: str,
) -> Dict[str, Any]:
    if mode not in {"prompted", "raw", "patched"}:
        raise ValueError(f"Unsupported eval mode: {mode}")
    rows = []
    correct = 0
    total_tokens = 0
    total_time = 0.0
    for index, example in enumerate(examples):
        user_text = task.contextual_prompt(example) if mode == "prompted" else example.raw_query
        pred = generate_answer(
            model,
            tokenizer,
            user_text=user_text,
            device=device,
            max_new_tokens=max_new_tokens,
            thought_patches=thought_patches if mode == "patched" else None,
        )
        answer_text = str(example.answer)
        is_correct = pred["normalized"] == answer_text
        correct += int(is_correct)
        total_tokens += pred["new_tokens"]
        total_time += pred["elapsed_s"]
        rows.append(
            {
                "index": index,
                "query": example.raw_query,
                "expected": answer_text,
                "prediction": pred["raw_text"],
                "normalized_prediction": pred["normalized"],
                "correct": is_correct,
                "latency_s": round(pred["elapsed_s"], 4),
                "new_tokens": pred["new_tokens"],
            }
        )
    accuracy = 100.0 * correct / max(len(examples), 1)
    avg_latency = total_time / max(len(examples), 1)
    avg_new_tokens = total_tokens / max(len(examples), 1)
    return {
        "mode": mode,
        "accuracy": accuracy,
        "avg_latency_s": avg_latency,
        "avg_new_tokens": avg_new_tokens,
        "rows": rows,
    }


def fit_one_example(
    *,
    model: Qwen3Model,
    tokenizer: Any,
    thought_patches,
    task: TaskSpec,
    example: ArithmeticExample,
    device: str,
    learning_rate: float,
    rho: float,
    alpha: float,
    patch_fc3: bool,
) -> None:
    answer_text = str(example.answer)
    ctx_ids, ctx_pos = build_alignment_positions(tokenizer, task.contextual_prompt(example), example.raw_query, answer_text)
    raw_ids, raw_pos = build_alignment_positions(tokenizer, example.raw_query, example.raw_query, answer_text)

    ctx_trace = capture_traces(model, ctx_ids, device=device, thought_patches=None)

    for layer_id, patch in enumerate(thought_patches):
        raw_trace = capture_traces(model, raw_ids, device=device, thought_patches=thought_patches)

        dz = ctx_trace[layer_id]["att_resid"][0, ctx_pos, :] - raw_trace[layer_id]["att_resid"][0, raw_pos, :]
        db = dz.mean(dim=0)
        a = raw_trace[layer_id]["mlp_in"][0, raw_pos, :]
        h = raw_trace[layer_id]["mlp_hidden"][0, raw_pos, :]
        d = raw_trace[layer_id]["mlp_out"][0, raw_pos, :]
        d_ctx = ctx_trace[layer_id]["mlp_out"][0, ctx_pos, :]

        block = model.trf_blocks[layer_id]
        w1 = (block.ff.fc1.weight.detach() + patch.d_fc1.detach().to(block.ff.fc1.weight.device, dtype=block.ff.fc1.weight.dtype)).float().cpu()
        w2 = (block.ff.fc2.weight.detach() + patch.d_fc2.detach().to(block.ff.fc2.weight.device, dtype=block.ff.fc2.weight.dtype)).float().cpu()
        target_fc1 = dz @ w1.T
        target_fc2 = dz @ w2.T
        target_fc3 = d_ctx - d

        d_fc1 = solve_weight_update(a, target_fc1, rho)
        d_fc2 = solve_weight_update(a, target_fc2, rho)
        scale = learning_rate * alpha
        patch.d_fc1.add_(scale * d_fc1.to(device=patch.d_fc1.device, dtype=patch.d_fc1.dtype))
        patch.d_fc2.add_(scale * d_fc2.to(device=patch.d_fc2.device, dtype=patch.d_fc2.dtype))
        if patch.d_bias is not None:
            patch.d_bias.add_(scale * db.to(device=patch.d_bias.device, dtype=patch.d_bias.dtype))
        if patch_fc3:
            d_fc3 = solve_weight_update(h, target_fc3, rho)
            patch.d_fc3.add_(scale * d_fc3.to(device=patch.d_fc3.device, dtype=patch.d_fc3.dtype))


def fit_layer_batch(
    *,
    model: Qwen3Model,
    tokenizer: Any,
    thought_patches,
    task: TaskSpec,
    examples: Sequence[ArithmeticExample],
    device: str,
    learning_rate: float,
    rho: float,
    alpha: float,
    patch_fc3: bool,
) -> None:
    prepared = []
    for example in examples:
        answer_text = str(example.answer)
        ctx_ids, ctx_pos = build_alignment_positions(tokenizer, task.contextual_prompt(example), example.raw_query, answer_text)
        raw_ids, raw_pos = build_alignment_positions(tokenizer, example.raw_query, example.raw_query, answer_text)
        ctx_trace = capture_traces(model, ctx_ids, device=device, thought_patches=None)
        prepared.append(
            {
                "example": example,
                "ctx_trace": ctx_trace,
                "raw_ids": raw_ids,
                "ctx_pos": ctx_pos,
                "raw_pos": raw_pos,
            }
        )

    scale = learning_rate * alpha
    for layer_id, patch in enumerate(thought_patches):
        a_bank = []
        h_bank = []
        dz_bank = []
        d_bank = []
        d_ctx_bank = []
        for item in prepared:
            raw_trace = capture_traces(model, item["raw_ids"], device=device, thought_patches=thought_patches)
            dz_bank.append(item["ctx_trace"][layer_id]["att_resid"][0, item["ctx_pos"], :] - raw_trace[layer_id]["att_resid"][0, item["raw_pos"], :])
            a_bank.append(raw_trace[layer_id]["mlp_in"][0, item["raw_pos"], :])
            h_bank.append(raw_trace[layer_id]["mlp_hidden"][0, item["raw_pos"], :])
            d_bank.append(raw_trace[layer_id]["mlp_out"][0, item["raw_pos"], :])
            d_ctx_bank.append(item["ctx_trace"][layer_id]["mlp_out"][0, item["ctx_pos"], :])

        dz = torch.cat(dz_bank, dim=0)
        a = torch.cat(a_bank, dim=0)
        h = torch.cat(h_bank, dim=0)
        d = torch.cat(d_bank, dim=0)
        d_ctx = torch.cat(d_ctx_bank, dim=0)
        db = dz.mean(dim=0)

        block = model.trf_blocks[layer_id]
        w1 = (block.ff.fc1.weight.detach() + patch.d_fc1.detach().to(block.ff.fc1.weight.device, dtype=block.ff.fc1.weight.dtype)).float().cpu()
        w2 = (block.ff.fc2.weight.detach() + patch.d_fc2.detach().to(block.ff.fc2.weight.device, dtype=block.ff.fc2.weight.dtype)).float().cpu()
        target_fc1 = dz @ w1.T
        target_fc2 = dz @ w2.T

        d_fc1 = solve_weight_update(a, target_fc1, rho)
        d_fc2 = solve_weight_update(a, target_fc2, rho)

        patch.d_fc1.add_(scale * d_fc1.to(device=patch.d_fc1.device, dtype=patch.d_fc1.dtype))
        patch.d_fc2.add_(scale * d_fc2.to(device=patch.d_fc2.device, dtype=patch.d_fc2.dtype))
        if patch.d_bias is not None:
            patch.d_bias.add_(scale * db.to(device=patch.d_bias.device, dtype=patch.d_bias.dtype))

        if patch_fc3:
            target_fc3 = d_ctx - d
            d_fc3 = solve_weight_update(h, target_fc3, rho)
            patch.d_fc3.add_(scale * d_fc3.to(device=patch.d_fc3.device, dtype=patch.d_fc3.dtype))


def metric_pm(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean_value:.2f} ± {std_value:.2f}"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_list = [str(header) for header in headers]
    row_list = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in header_list]
    for row in row_list:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(values: Sequence[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    parts = [sep, fmt_row(header_list), sep]
    for row in row_list:
        parts.append(fmt_row(row))
    parts.append(sep)
    return "\n".join(parts)


def example_preview(examples: Sequence[ArithmeticExample], limit: int = 3) -> str:
    preview = []
    for example in examples[:limit]:
        preview.append(f"{example.raw_query} {example.answer}")
    return "; ".join(preview)


def task_seed_run(
    *,
    model: Qwen3Model,
    tokenizer: Any,
    thought_patches,
    task: TaskSpec,
    dataset: Dict[str, List[ArithmeticExample]],
    device: str,
    learning_rate: float,
    rho: float,
    alpha: float,
    max_new_tokens: int,
    eval_every_step: bool,
    fit_mode: str,
    patch_fc3: bool,
    seed_index: int,
) -> Dict[str, Any]:
    reset_patches(thought_patches)

    train_examples = dataset["train"]
    eval_examples = dataset["eval"]
    LOGGER.info(
        "[%s seed=%d] train=%d eval=%d train_preview=%s",
        task.slug,
        seed_index,
        len(train_examples),
        len(eval_examples),
        example_preview(train_examples),
    )

    baseline_prompted = evaluate_examples(
        model,
        tokenizer,
        thought_patches,
        eval_examples,
        task,
        device=device,
        max_new_tokens=max_new_tokens,
        mode="prompted",
    )
    baseline_raw = evaluate_examples(
        model,
        tokenizer,
        thought_patches,
        eval_examples,
        task,
        device=device,
        max_new_tokens=max_new_tokens,
        mode="raw",
    )
    LOGGER.info(
        "[%s seed=%d] baseline eval prompted=%.2f%% raw=%.2f%%",
        task.slug,
        seed_index,
        baseline_prompted["accuracy"],
        baseline_raw["accuracy"],
    )

    step_history: List[Dict[str, Any]] = []
    best_eval = 0.0
    fit_start = time.perf_counter()
    for step, example in enumerate(train_examples, start=1):
        step_start = time.perf_counter()
        if fit_mode == "sequential":
            fit_one_example(
                model=model,
                tokenizer=tokenizer,
                thought_patches=thought_patches,
                task=task,
                example=example,
                device=device,
                learning_rate=learning_rate,
                rho=rho,
                alpha=alpha,
                patch_fc3=patch_fc3,
            )
        elif fit_mode == "layer_batch":
            fit_layer_batch(
                model=model,
                tokenizer=tokenizer,
                thought_patches=thought_patches,
                task=task,
                examples=train_examples[:step],
                device=device,
                learning_rate=learning_rate / step,
                rho=rho,
                alpha=alpha,
                patch_fc3=patch_fc3,
            )
        else:
            raise ValueError(f"Unsupported fit_mode: {fit_mode}")
        fit_elapsed = time.perf_counter() - step_start
        norms = patch_norms(thought_patches)

        step_record: Dict[str, Any] = {
            "step": step,
            "train_example": example.raw_query,
            "expected": str(example.answer),
            "fit_elapsed_s": fit_elapsed,
            "fc1_norm": norms["fc1"],
            "fc2_norm": norms["fc2"],
            "fc3_norm": norms["fc3"],
            "bias_norm": norms["bias"],
        }

        if eval_every_step:
            patched_train = evaluate_examples(
                model,
                tokenizer,
                thought_patches,
                train_examples,
                task,
                device=device,
                max_new_tokens=max_new_tokens,
                mode="patched",
            )
            patched_eval = evaluate_examples(
                model,
                tokenizer,
                thought_patches,
                eval_examples,
                task,
                device=device,
                max_new_tokens=max_new_tokens,
                mode="patched",
            )
            best_eval = max(best_eval, patched_eval["accuracy"])
            step_record["patched_train_accuracy"] = patched_train["accuracy"]
            step_record["patched_eval_accuracy"] = patched_eval["accuracy"]
            step_record["best_eval_accuracy"] = best_eval
            LOGGER.info(
                "[%s seed=%d step=%02d/%02d] patched train=%.2f%% eval=%.2f%% best=%.2f%% norms(fc1=%.2f fc2=%.2f fc3=%.2f bias=%.2f) fit=%.2fs",
                task.slug,
                seed_index,
                step,
                len(train_examples),
                patched_train["accuracy"],
                patched_eval["accuracy"],
                best_eval,
                norms["fc1"],
                norms["fc2"],
                norms["fc3"],
                norms["bias"],
                fit_elapsed,
            )
        step_history.append(step_record)

    total_fit_time = time.perf_counter() - fit_start
    final_patched_train = evaluate_examples(
        model,
        tokenizer,
        thought_patches,
        train_examples,
        task,
        device=device,
        max_new_tokens=max_new_tokens,
        mode="patched",
    )
    final_patched_eval = evaluate_examples(
        model,
        tokenizer,
        thought_patches,
        eval_examples,
        task,
        device=device,
        max_new_tokens=max_new_tokens,
        mode="patched",
    )
    best_eval = max(best_eval, final_patched_eval["accuracy"])
    LOGGER.info(
        "[%s seed=%d] final patched train=%.2f%% eval=%.2f%% best_eval=%.2f%% fit_time=%.1fs",
        task.slug,
        seed_index,
        final_patched_train["accuracy"],
        final_patched_eval["accuracy"],
        best_eval,
        total_fit_time,
    )

    step_rows = [
        [
            record["step"],
            f"{record.get('patched_train_accuracy', float('nan')):.2f}" if "patched_train_accuracy" in record else "-",
            f"{record.get('patched_eval_accuracy', float('nan')):.2f}" if "patched_eval_accuracy" in record else "-",
            f"{record.get('best_eval_accuracy', float('nan')):.2f}" if "best_eval_accuracy" in record else "-",
            f"{record['fc1_norm']:.2f}",
            f"{record['fc2_norm']:.2f}",
            f"{record['fc3_norm']:.2f}",
            f"{record['bias_norm']:.2f}",
            f"{record['fit_elapsed_s']:.2f}",
        ]
        for record in step_history
    ]
    print()
    print(f"Step metrics for task={task.slug} seed={seed_index}")
    print(
        render_table(
            ["Step", "Patched train", "Patched eval", "Best eval", "||d_fc1||", "||d_fc2||", "||d_fc3||", "||d_bias||", "Fit s"],
            step_rows,
        )
    )

    return {
        "task": task.slug,
        "task_title": task.title,
        "seed": seed_index,
        "train_examples": [asdict(example) for example in train_examples],
        "eval_examples": [asdict(example) for example in eval_examples],
        "baseline_prompted_eval": baseline_prompted,
        "baseline_raw_eval": baseline_raw,
        "final_patched_train": final_patched_train,
        "final_patched_eval": final_patched_eval,
        "best_patched_eval_accuracy": best_eval,
        "fit_time_s": total_fit_time,
        "step_history": step_history,
        "patch_norms": patch_norms(thought_patches),
    }


def build_final_tables(results: Sequence[Dict[str, Any]]) -> tuple[str, str]:
    seed_rows = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["task"], []).append(result)
        seed_rows.append(
            [
                result["task_title"],
                result["seed"],
                f"{result['baseline_prompted_eval']['accuracy']:.2f}%",
                f"{result['baseline_raw_eval']['accuracy']:.2f}%",
                f"{result['final_patched_eval']['accuracy']:.2f}%",
                f"{result['best_patched_eval_accuracy']:.2f}%",
                f"{result['fit_time_s']:.1f}",
            ]
        )

    aggregate_rows = []
    for task_slug, task_results in grouped.items():
        title = task_results[0]["task_title"]
        prompted_values = [row["baseline_prompted_eval"]["accuracy"] for row in task_results]
        raw_values = [row["baseline_raw_eval"]["accuracy"] for row in task_results]
        patched_values = [row["final_patched_eval"]["accuracy"] for row in task_results]
        aggregate_rows.append(
            [
                title,
                metric_pm(prompted_values),
                metric_pm(raw_values),
                metric_pm(patched_values),
            ]
        )

    seed_table = render_table(
        ["Task", "Seed", "Original w/ context", "Original w/o context", "Patched w/o context", "Best patched", "Fit time s"],
        seed_rows,
    )
    aggregate_table = render_table(
        ["Task", "Original model w/ context", "Original model w/o context", "Patched model w/o context"],
        aggregate_rows,
    )
    return seed_table, aggregate_table


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown_tasks = [task for task in task_names if task not in TASKS]
    if unknown_tasks:
        raise SystemExit(f"Unknown tasks: {', '.join(unknown_tasks)}")

    LOGGER.info(
        "Starting thought-patching benchmark: tasks=%s seeds=%d train=%d eval=%d device=%s dtype=%s",
        task_names,
        args.seeds,
        args.train_examples,
        args.eval_examples,
        device,
        str(dtype).replace("torch.", ""),
    )

    model, tokenizer, repo_id, local_dir = build_model(
        model_size=args.model_size,
        model_type=args.model_type,
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        device=device,
        dtype=dtype,
    )
    thought_patches = build_empty_thought_patches(model)

    all_results: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()
    for seed_index in range(args.seeds):
        for task_name in task_names:
            task = TASKS[task_name]
            dataset = make_examples(
                task_name,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                train_examples=args.train_examples,
                eval_examples=args.eval_examples,
                seed=seed_index,
                digit_min=args.digit_min,
                digit_max=args.digit_max,
                paper_filter=args.paper_filter,
            )
            result = task_seed_run(
                model=model,
                tokenizer=tokenizer,
                thought_patches=thought_patches,
                task=task,
                dataset=dataset,
                device=device,
                learning_rate=args.learning_rate,
                rho=args.rho,
                alpha=args.alpha,
                max_new_tokens=args.max_new_tokens,
                eval_every_step=args.eval_every_step,
                fit_mode=args.fit_mode,
                patch_fc3=args.patch_fc3,
                seed_index=seed_index,
            )
            all_results.append(result)

    seed_table, aggregate_table = build_final_tables(all_results)
    total_elapsed = time.perf_counter() - wall_start

    print()
    print("Per-seed summary")
    print(seed_table)
    print()
    print("Paper-style summary")
    print(aggregate_table)

    serializable_args = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable_args[key] = str(value)
        else:
            serializable_args[key] = value

    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **serializable_args,
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "repo_id": repo_id,
            "local_dir": local_dir,
            "tasks": task_names,
        },
        "wall_time_s": total_elapsed,
        "results": all_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    LOGGER.info("Saved metrics artifact to %s", args.out)
    LOGGER.info("Total wall time: %.1fs", total_elapsed)


if __name__ == "__main__":
    main()
