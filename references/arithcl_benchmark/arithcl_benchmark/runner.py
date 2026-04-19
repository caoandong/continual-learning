from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from .core import BenchmarkResults, Example, Scenario, StageConfig
from .utils import accuracy


class BenchmarkRunner:
    def __init__(self, scenario: Scenario, base_seed: int = 0):
        self.scenario = scenario
        self.base_seed = base_seed

    def _eval_sets(self) -> Dict[str, Dict[str, List[Example]]]:
        prepared: Dict[str, Dict[str, List[Example]]] = {}
        for stage_idx, stage in enumerate(self.scenario.stages):
            task_name = stage.task.name
            if task_name in prepared:
                continue
            prepared[task_name] = {}
            for split_eval in stage.eval_splits:
                rng = random.Random(self.base_seed + 10_000 * stage_idx + hash((task_name, split_eval.split_name)) % 1000)
                prepared[task_name][split_eval.split_name] = [
                    stage.task.sample(split_eval.split_name, rng) for _ in range(split_eval.num_examples)
                ]
        return prepared

    @staticmethod
    def _evaluate_model(model, eval_sets: Dict[str, Dict[str, List[Example]]]) -> Dict[str, Dict[str, float]]:
        matrix: Dict[str, Dict[str, float]] = {}
        for task_name, split_map in eval_sets.items():
            matrix[task_name] = {}
            for split_name, examples in split_map.items():
                preds = model.predict(examples)
                gold = [ex.answer for ex in examples]
                matrix[task_name][split_name] = accuracy(preds, gold)
        return matrix

    def _train_stage(
        self,
        model,
        stage: StageConfig,
        eval_sets: Dict[str, Dict[str, List[Example]]],
    ) -> Dict[str, Dict[int, float]]:
        checkpoints = stage.normalized_checkpoints()
        curves: Dict[str, Dict[int, float]] = {split.split_name: {} for split in stage.eval_splits}
        task_name = stage.task.name

        # zero-shot checkpoint
        for split_eval in stage.eval_splits:
            examples = eval_sets[task_name][split_eval.split_name]
            preds = model.predict(examples)
            curves[split_eval.split_name][0] = accuracy(preds, [ex.answer for ex in examples])

        rng = random.Random(self.base_seed + stage.train_seed + hash(stage.name) % 9973)
        seen = 0
        while seen < stage.train_examples:
            batch = [
                stage.task.sample("id", rng)
                for _ in range(min(stage.batch_size, stage.train_examples - seen))
            ]
            model.update(batch)
            seen += len(batch)
            if seen in checkpoints:
                for split_eval in stage.eval_splits:
                    examples = eval_sets[task_name][split_eval.split_name]
                    preds = model.predict(examples)
                    curves[split_eval.split_name][seen] = accuracy(preds, [ex.answer for ex in examples])

        # ensure final checkpoint is present
        final_seen = stage.train_examples
        for split_eval in stage.eval_splits:
            if final_seen not in curves[split_eval.split_name]:
                examples = eval_sets[task_name][split_eval.split_name]
                preds = model.predict(examples)
                curves[split_eval.split_name][final_seen] = accuracy(preds, [ex.answer for ex in examples])
        return curves

    def run(self, make_model: Callable[[], object]) -> BenchmarkResults:
        eval_sets = self._eval_sets()

        continual_model = make_model()
        initial_matrix = self._evaluate_model(continual_model, eval_sets)
        stage_final_matrices: List[Dict[str, Dict[str, float]]] = []
        adaptation_curves: Dict[str, Dict[str, Dict[int, float]]] = {}
        scratch_curves: Dict[str, Dict[str, Dict[int, float]]] = {}

        for stage in self.scenario.stages:
            adaptation_curves[stage.name] = self._train_stage(continual_model, stage, eval_sets)
            stage_final_matrices.append(self._evaluate_model(continual_model, eval_sets))

            scratch_model = make_model()
            scratch_curves[stage.name] = self._train_stage(scratch_model, stage, eval_sets)

        metadata = {
            "scenario_description": self.scenario.description,
            "base_seed": self.base_seed,
            "stage_order": [stage.name for stage in self.scenario.stages],
            "tasks": [stage.task.name for stage in self.scenario.stages],
        }
        return BenchmarkResults(
            scenario_name=self.scenario.name,
            initial_matrix=initial_matrix,
            stage_final_matrices=stage_final_matrices,
            adaptation_curves=adaptation_curves,
            scratch_curves=scratch_curves,
            metadata=metadata,
        )

    @staticmethod
    def save_results(results: BenchmarkResults, out_path: str | Path) -> None:
        path = Path(out_path)
        path.write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")
