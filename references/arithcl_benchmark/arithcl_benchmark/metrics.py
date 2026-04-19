from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .core import BenchmarkResults, Scenario


def _stage_task_names(scenario: Scenario) -> List[str]:
    return [stage.task.name for stage in scenario.stages]


def _composition_indices(scenario: Scenario) -> List[int]:
    idxs = []
    for i, stage in enumerate(scenario.stages):
        if len(set(stage.task.skill_dependencies())) > 1:
            idxs.append(i)
    return idxs


def _examples_to_threshold(curve: Mapping[int, float], threshold: float) -> int | None:
    for seen in sorted(curve):
        if curve[seen] >= threshold:
            return seen
    return None


def summarize_results(
    results: BenchmarkResults,
    scenario: Scenario,
    threshold_split: str = "ood_length",
    threshold: float = 0.9,
) -> Dict[str, Any]:
    stage_task_names = _stage_task_names(scenario)
    final_matrix = results.stage_final_matrices[-1]
    initial = results.initial_matrix
    per_split = {}

    split_names = list(next(iter(final_matrix.values())).keys())
    for split_name in split_names:
        final_avg = sum(final_matrix[task][split_name] for task in stage_task_names) / len(stage_task_names)

        # FWT: performance on task before training it minus initial baseline.
        fwt_vals = []
        for j in range(1, len(stage_task_names)):
            task_name = stage_task_names[j]
            before_training_matrix = results.stage_final_matrices[j - 1]
            fwt_vals.append(before_training_matrix[task_name][split_name] - initial[task_name][split_name])
        fwt = sum(fwt_vals) / len(fwt_vals) if fwt_vals else 0.0

        # Final BWT: final performance minus performance immediately after task was learned.
        bwt_vals = []
        forgetting_vals = []
        for j, task_name in enumerate(stage_task_names[:-1]):
            learned_matrix = results.stage_final_matrices[j]
            final_score = final_matrix[task_name][split_name]
            bwt_vals.append(final_score - learned_matrix[task_name][split_name])
            best_score = max(m[task_name][split_name] for m in results.stage_final_matrices[j:])
            forgetting_vals.append(best_score - final_score)
        final_bwt = sum(bwt_vals) / len(bwt_vals) if bwt_vals else 0.0
        max_forgetting = sum(forgetting_vals) / len(forgetting_vals) if forgetting_vals else 0.0

        # zero-shot composition score: accuracy on composition tasks before direct training
        composition_scores = []
        for idx in _composition_indices(scenario):
            task_name = stage_task_names[idx]
            if idx == 0:
                continue
            before_training_matrix = results.stage_final_matrices[idx - 1]
            composition_scores.append(before_training_matrix[task_name][split_name])
        zero_shot_composition = (
            sum(composition_scores) / len(composition_scores) if composition_scores else 0.0
        )

        per_split[split_name] = {
            "final_average_accuracy": round(final_avg, 4),
            "forward_transfer": round(fwt, 4),
            "final_backward_transfer": round(final_bwt, 4),
            "mean_max_forgetting": round(max_forgetting, 4),
            "zero_shot_composition": round(zero_shot_composition, 4),
        }

    threshold_summary: Dict[str, Dict[str, Any]] = {}
    for stage in scenario.stages:
        continual_curve = results.adaptation_curves[stage.name][threshold_split]
        scratch_curve = results.scratch_curves[stage.name][threshold_split]
        continual_n = _examples_to_threshold(continual_curve, threshold)
        scratch_n = _examples_to_threshold(scratch_curve, threshold)
        gain = None
        if continual_n is not None and scratch_n is not None:
            if continual_n == 0 and scratch_n == 0:
                gain = 1.0
            elif continual_n == 0:
                gain = "inf"
            elif scratch_n != 0:
                gain = round(scratch_n / continual_n, 4)
        threshold_summary[stage.name] = {
            "threshold_split": threshold_split,
            "threshold": threshold,
            "continual_examples_to_threshold": continual_n,
            "scratch_examples_to_threshold": scratch_n,
            "transfer_efficiency_gain": gain,
        }

    return {
        "scenario_name": results.scenario_name,
        "per_split": per_split,
        "threshold_summary": threshold_summary,
        "metadata": results.metadata,
    }
