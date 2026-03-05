from __future__ import annotations

import argparse
import logging

from continual_learning.constants import DEFAULT_LAYER_SIZES, DEFAULT_MODEL
from continual_learning.experiment import evaluate_sample, train_on_sample
from continual_learning.llm import create_batch_llm_caller
from continual_learning.network import create_network_state
from continual_learning.types import ExperimentOptions


def configure_logging(*, verbose: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    for noisy in ("httpx", "LiteLLM", "litellm", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_layer_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal continual-learning smoke test",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="LLM model name (default: %(default)s)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock LLM (no API key needed)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug-level logging",
    )
    parser.add_argument(
        "--layer-sizes",
        type=parse_layer_sizes,
        default=DEFAULT_LAYER_SIZES,
        help="Comma-separated layer sizes (default: 2,1)",
    )
    parser.add_argument(
        "--sample",
        default="alpha beta gamma",
        help="Training sample tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--target-label",
        default="class_a",
        help="Target label for the training sample (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-sample",
        default="gamma beta alpha",
        help="Evaluation sample tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Training rounds to run before evaluation (default: %(default)s)",
    )
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    options = ExperimentOptions(
        model=args.model,
        use_mock=args.mock,
        layer_sizes=args.layer_sizes,
        verbose=args.verbose,
    )
    call_llm_batch = create_batch_llm_caller(options)
    state = create_network_state(args.layer_sizes)
    for _ in range(args.rounds):
        state = train_on_sample(state, args.sample, args.target_label, call_llm_batch)
    _, prediction = evaluate_sample(state, args.eval_sample, call_llm_batch)
    logging.getLogger(__name__).info(
        "sample=%s target=%s eval_sample=%s prediction=%s",
        args.sample,
        args.target_label,
        args.eval_sample,
        prediction,
    )


if __name__ == "__main__":
    main()
