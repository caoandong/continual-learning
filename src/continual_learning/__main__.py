from __future__ import annotations

import argparse
import logging
from pathlib import Path

from continual_learning.constants import DEFAULT_LAYER_SIZES, DEFAULT_MODEL
from continual_learning.experiment import run_experiment
from continual_learning.llm import create_llm_caller
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
        description="Semantic Predictive Cellular Automaton — continual learning demo",
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
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    options = ExperimentOptions(
        model=args.model,
        use_mock=args.mock,
        layer_sizes=args.layer_sizes,
        verbose=args.verbose,
    )
    call_llm = create_llm_caller(options)
    run_experiment(options, call_llm)


if __name__ == "__main__":
    main()
