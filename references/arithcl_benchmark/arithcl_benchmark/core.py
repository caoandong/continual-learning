from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    task_name: str
    split_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitEvaluation:
    split_name: str
    num_examples: int


@dataclass(frozen=True)
class StageConfig:
    name: str
    task: Any
    train_examples: int = 1024
    batch_size: int = 32
    checkpoints: Sequence[int] = (0, 128, 256, 512, 1024)
    eval_splits: Sequence[SplitEvaluation] = (
        SplitEvaluation("id", 256),
        SplitEvaluation("ood_length", 256),
        SplitEvaluation("ood_template", 256),
    )
    train_seed: int = 0

    def normalized_checkpoints(self) -> List[int]:
        uniq = sorted(set(int(x) for x in self.checkpoints if 0 <= int(x) <= self.train_examples))
        if 0 not in uniq:
            uniq = [0] + uniq
        if self.train_examples not in uniq:
            uniq.append(self.train_examples)
        return uniq


@dataclass(frozen=True)
class Scenario:
    name: str
    stages: Sequence[StageConfig]
    description: str = ""


@dataclass
class BenchmarkResults:
    scenario_name: str
    initial_matrix: Dict[str, Dict[str, float]]
    stage_final_matrices: List[Dict[str, Dict[str, float]]]
    adaptation_curves: Dict[str, Dict[str, Dict[int, float]]]
    scratch_curves: Dict[str, Dict[str, Dict[int, float]]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
