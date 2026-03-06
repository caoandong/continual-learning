#!/usr/bin/env python3
"""Run the continual-learning benchmark on literal MNIST pixel streams."""
from __future__ import annotations

import argparse
import gzip
import logging
import struct
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from continual_learning.constants import DEFAULT_LAYER_SIZES, DEFAULT_MODEL
from continual_learning.experiment import evaluate_sample, train_on_sample
from continual_learning.llm import create_batch_llm_caller
from continual_learning.network import create_network_state
from continual_learning.types import BatchLlmCaller, ExperimentOptions, NetworkState

logger = logging.getLogger(__name__)

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train": (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    "test": (
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ),
}
DEFAULT_DATASET_DIR = Path(".cache/mnist")


@dataclass(frozen=True)
class MnistExample:
    index: int
    label: int
    raw_stream: str


@dataclass(frozen=True)
class BenchmarkConfig:
    digits: tuple[int, ...]
    rounds: int
    layer_sizes: tuple[int, ...]
    train_examples_per_digit: int
    eval_examples_per_digit: int
    dataset_dir: Path


def configure_logging(*, verbose: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    for noisy in ("httpx", "LiteLLM", "litellm", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_digits(value: str) -> tuple[int, ...]:
    digits = tuple(int(item.strip()) for item in value.split(","))
    if not digits:
        raise argparse.ArgumentTypeError("at least one digit is required")
    for digit in digits:
        if digit < 0 or digit > 9:
            raise argparse.ArgumentTypeError("digits must be in the range 0..9")
    return digits


def parse_layer_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(","))


def dataset_file_paths(*, split: str, dataset_dir: Path) -> tuple[Path, Path]:
    image_name, label_name = MNIST_FILES[split]
    return dataset_dir / image_name, dataset_dir / label_name


def ensure_file(*, destination: Path, url: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, destination)  # noqa: S310


def ensure_dataset_files(*, split: str, dataset_dir: Path) -> tuple[Path, Path]:
    image_path, label_path = dataset_file_paths(split=split, dataset_dir=dataset_dir)
    ensure_file(destination=image_path, url=MNIST_BASE_URL + image_path.name)
    ensure_file(destination=label_path, url=MNIST_BASE_URL + label_path.name)
    return image_path, label_path


def read_gzip_bytes(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def image_layout(image_payload: bytes) -> tuple[int, int]:
    magic, count, rows, columns = struct.unpack(">IIII", image_payload[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected image magic number: {magic}")
    return count, rows * columns


def label_count(label_payload: bytes) -> int:
    magic, count = struct.unpack(">II", label_payload[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected label magic number: {magic}")
    return count


def pixel_stream(pixel_bytes: bytes) -> str:
    return " ".join(str(value) for value in pixel_bytes)


def collect_examples(
    *,
    split: str,
    digits: tuple[int, ...],
    per_digit: int,
    dataset_dir: Path,
) -> dict[int, tuple[MnistExample, ...]]:
    image_path, label_path = ensure_dataset_files(split=split, dataset_dir=dataset_dir)
    image_payload = read_gzip_bytes(image_path)
    label_payload = read_gzip_bytes(label_path)
    image_count, image_size = image_layout(image_payload)
    if image_count != label_count(label_payload):
        raise ValueError("Image/label count mismatch in MNIST split")

    selected: dict[int, list[MnistExample]] = {digit: [] for digit in digits}
    label_bytes = label_payload[8:]
    image_bytes = memoryview(image_payload)[16:]
    for index, label in enumerate(label_bytes):
        if label not in selected or len(selected[label]) >= per_digit:
            continue
        start = index * image_size
        stop = start + image_size
        stream = pixel_stream(image_bytes[start:stop].tobytes())
        selected[label].append(MnistExample(index=index, label=label, raw_stream=stream))
        if all(len(selected[digit]) >= per_digit for digit in digits):
            break

    missing = [digit for digit, examples in selected.items() if len(examples) < per_digit]
    if missing:
        raise ValueError(f"Could not collect enough MNIST examples for digits: {missing}")
    return {digit: tuple(examples) for digit, examples in selected.items()}


def preview_stream(stream: str) -> str:
    values = stream.split()
    preview = " ".join(values[:12])
    return f"{preview} ... ({len(values)} values)"


def evaluate_examples(
    *,
    state: NetworkState,
    digits: Sequence[int],
    examples_by_digit: dict[int, tuple[MnistExample, ...]],
    call_llm_batch: BatchLlmCaller,
) -> tuple[NetworkState, int, int]:
    correct = 0
    total = 0
    current = state
    for digit in digits:
        for example in examples_by_digit[digit]:
            current, prediction = evaluate_sample(current, example.raw_stream, call_llm_batch)
            is_correct = prediction == str(digit)
            if is_correct:
                correct += 1
            total += 1
            status = "correct" if is_correct else "WRONG"
            logger.info(
                "  digit %d | sample %d | pred: %-12s | %s | %s",
                digit,
                example.index,
                prediction,
                status,
                preview_stream(example.raw_stream),
            )
    return current, correct, total


def train_digit(
    *,
    state: NetworkState,
    examples: tuple[MnistExample, ...],
    rounds: int,
    call_llm_batch: BatchLlmCaller,
) -> NetworkState:
    current = state
    for round_number in range(1, rounds + 1):
        for example in examples:
            current = train_on_sample(
                current,
                example.raw_stream,
                str(example.label),
                call_llm_batch,
            )
            _, prediction = evaluate_sample(current, example.raw_stream, call_llm_batch)
            status = "correct" if prediction == str(example.label) else "wrong"
            logger.info(
                "  round %d/%d | digit %d | sample %d | pred: %-12s | %s",
                round_number,
                rounds,
                example.label,
                example.index,
                prediction,
                status,
            )
    return current


def inspect_neuron_states(state: NetworkState) -> None:
    for layer in state.layers:
        for neuron in layer.neurons:
            logger.info("  %s: %s", neuron.name, neuron.state)


def run_benchmark(*, config: BenchmarkConfig, call_llm_batch: BatchLlmCaller) -> None:
    train_examples = collect_examples(
        split="train",
        digits=config.digits,
        per_digit=config.train_examples_per_digit,
        dataset_dir=config.dataset_dir,
    )
    eval_examples = collect_examples(
        split="test",
        digits=config.digits,
        per_digit=config.eval_examples_per_digit,
        dataset_dir=config.dataset_dir,
    )
    state = create_network_state(config.layer_sizes)

    logger.info(
        "Network: layers=%s digits=%s rounds=%d train_per_digit=%d eval_per_digit=%d",
        config.layer_sizes,
        config.digits,
        config.rounds,
        config.train_examples_per_digit,
        config.eval_examples_per_digit,
    )
    logger.info("=== Initial evaluation on held-out MNIST examples ===")
    _, correct, total = evaluate_examples(
        state=state,
        digits=config.digits,
        examples_by_digit=eval_examples,
        call_llm_batch=call_llm_batch,
    )
    logger.info("  initial accuracy: %d/%d", correct, total)

    learned_so_far: list[int] = []
    for digit in config.digits:
        learned_so_far.append(digit)
        logger.info("--- Training on digit %d ---", digit)
        state = train_digit(
            state=state,
            examples=train_examples[digit],
            rounds=config.rounds,
            call_llm_batch=call_llm_batch,
        )
        logger.info("--- Evaluation on learned digits %s ---", learned_so_far)
        state, correct, total = evaluate_examples(
            state=state,
            digits=learned_so_far,
            examples_by_digit=eval_examples,
            call_llm_batch=call_llm_batch,
        )
        logger.info("  accuracy: %d/%d", correct, total)

    logger.info("=== Final evaluation on digits %s ===", config.digits)
    state, correct, total = evaluate_examples(
        state=state,
        digits=config.digits,
        examples_by_digit=eval_examples,
        call_llm_batch=call_llm_batch,
    )
    logger.info("  final accuracy: %d/%d", correct, total)
    logger.info("=== Learned neuron states ===")
    inspect_neuron_states(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual-learning benchmark on raw MNIST streams")
    parser.add_argument(
        "--digits",
        type=parse_digits,
        default=(0, 1, 2),
        help="Comma-separated digits to learn incrementally (default: 0,1,2)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Training rounds per digit (default: 1)",
    )
    parser.add_argument(
        "--layer-sizes",
        type=parse_layer_sizes,
        default=DEFAULT_LAYER_SIZES,
        help="Comma-separated layer sizes (default: 1,1)",
    )
    parser.add_argument(
        "--train-examples-per-digit",
        type=int,
        default=3,
        help="Number of training examples to use per digit (default: 3)",
    )
    parser.add_argument(
        "--eval-examples-per-digit",
        type=int,
        default=3,
        help="Number of held-out test examples to evaluate per digit (default: 3)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory used to cache MNIST files (default: .cache/mnist)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model name (default: %(default)s)",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    use_mock = args.mock or "--model" not in sys.argv
    options = ExperimentOptions(
        model=args.model,
        use_mock=use_mock,
        layer_sizes=args.layer_sizes,
        verbose=args.verbose,
    )
    config = BenchmarkConfig(
        digits=args.digits,
        rounds=args.rounds,
        layer_sizes=args.layer_sizes,
        train_examples_per_digit=args.train_examples_per_digit,
        eval_examples_per_digit=args.eval_examples_per_digit,
        dataset_dir=args.dataset_dir,
    )
    call_llm_batch = create_batch_llm_caller(options)
    run_benchmark(config=config, call_llm_batch=call_llm_batch)


if __name__ == "__main__":
    main()
