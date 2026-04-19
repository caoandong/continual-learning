from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import BlankLearner, ComposableSkillOracle, StageSkillOracle
from .metrics import summarize_results
from .runner import BenchmarkRunner
from .scenarios import extended_arith_v1, three_stage_arith_v1, warmup_transfer_v1

SCENARIOS = {
    "three_stage_arith_v1": three_stage_arith_v1,
    "extended_arith_v1": extended_arith_v1,
    "warmup_transfer_v1": warmup_transfer_v1,
}

BASELINES = {
    "blank": BlankLearner,
    "stage_oracle": StageSkillOracle,
    "composable_oracle": ComposableSkillOracle,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Arithmetic continual-learning benchmark")
    parser.add_argument("--scenario", default="three_stage_arith_v1", choices=SCENARIOS.keys())
    parser.add_argument("--baseline", default="composable_oracle", choices=BASELINES.keys())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-json", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]()
    runner = BenchmarkRunner(scenario=scenario, base_seed=args.seed)
    results = runner.run(BASELINES[args.baseline])
    summary = summarize_results(results, scenario)

    if args.results_json:
        Path(args.results_json).write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
