from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .core import Example
from .utils import normalize_answer, safe_eval_arithmetic


class BaseLearner:
    def update(self, batch: Sequence[Example]) -> None:
        raise NotImplementedError

    def predict(self, batch: Sequence[Example]) -> List[str]:
        raise NotImplementedError


class BlankLearner(BaseLearner):
    def update(self, batch: Sequence[Example]) -> None:
        return None

    def predict(self, batch: Sequence[Example]) -> List[str]:
        return ["" for _ in batch]


@dataclass
class StageSkillOracle(BaseLearner):
    learned_tasks: set[str] = field(default_factory=set)

    def update(self, batch: Sequence[Example]) -> None:
        for ex in batch:
            self.learned_tasks.add(ex.task_name)

    def predict(self, batch: Sequence[Example]) -> List[str]:
        outputs: List[str] = []
        for ex in batch:
            if ex.task_name in self.learned_tasks:
                outputs.append(ex.answer)
            else:
                outputs.append("")
        return outputs


@dataclass
class ComposableSkillOracle(BaseLearner):
    learned_skills: set[str] = field(default_factory=set)

    def update(self, batch: Sequence[Example]) -> None:
        for ex in batch:
            deps = set(ex.metadata.get("dependencies", []))
            self.learned_skills |= deps

    def predict(self, batch: Sequence[Example]) -> List[str]:
        outputs: List[str] = []
        for ex in batch:
            deps = set(ex.metadata.get("dependencies", []))
            outputs.append(ex.answer if deps.issubset(self.learned_skills) else "")
        return outputs
