from __future__ import annotations

import json
from pathlib import Path

from references.arithcl_benchmark.arithcl_benchmark.baselines import BlankLearner, ComposableSkillOracle, StageSkillOracle
from references.arithcl_benchmark.arithcl_benchmark.metrics import summarize_results
from references.arithcl_benchmark.arithcl_benchmark.runner import BenchmarkRunner
from references.arithcl_benchmark.arithcl_benchmark.scenarios import extended_arith_v1

scenario = extended_arith_v1()
runner = BenchmarkRunner(scenario, base_seed=123)

results_dir = Path("sample_results")
results_dir.mkdir(exist_ok=True)

for name, factory in [
    ("blank", BlankLearner),
    ("stage_oracle", StageSkillOracle),
    ("composable_oracle", ComposableSkillOracle),
]:
    results = runner.run(factory)
    summary = summarize_results(results, scenario, threshold_split="ood_length", threshold=0.9)
    (results_dir / f"{name}_results.json").write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
    (results_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {name}")
