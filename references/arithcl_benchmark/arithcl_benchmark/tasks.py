from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .core import Example
from .utils import encode_base, sample_int_with_digits, safe_eval_arithmetic


@dataclass(frozen=True)
class TaskSpec:
    min_digits: int = 1
    max_digits: int = 3
    min_operands: int = 2
    max_operands: int = 3
    max_terms: int | None = None
    max_depth: int | None = None
    alphabet_size: int | None = None
    input_length: int | None = None
    state_length: int | None = None
    steps: int | None = None
    templates: Sequence[str] = ()


class BaseTask:
    name = "base"
    primitive_skills: Sequence[str] = ()
    train_templates: Sequence[str] = ("Task: {task}\nInput: {expr}\nAnswer:",)
    ood_templates: Sequence[str] = ("Solve {task}: {expr}\nFinal answer:",)

    split_specs: Mapping[str, TaskSpec] = {}

    def sample(self, split_name: str, rng: random.Random) -> Example:
        raise NotImplementedError

    def skill_dependencies(self) -> Sequence[str]:
        return tuple(self.primitive_skills)

    def _choose_template(self, split_name: str, rng: random.Random) -> str:
        spec = self.split_specs[split_name]
        templates = spec.templates or (self.ood_templates if split_name == "ood_template" else self.train_templates)
        return rng.choice(list(templates))


class AdditionTask(BaseTask):
    name = "addition"
    primitive_skills = ("add",)
    train_templates = (
        "Task: add the numbers.\nInput: {expr}\nAnswer:",
        "Compute the sum.\n{expr} =",
    )
    ood_templates = (
        "What is the value of {expr}?\nReturn digits only.",
        "Add carefully: {expr}\nResult:",
    )
    split_specs = {
        "id": TaskSpec(min_digits=2, max_digits=4, min_operands=2, max_operands=3),
        "ood_length": TaskSpec(min_digits=5, max_digits=8, min_operands=3, max_operands=5),
        "ood_template": TaskSpec(min_digits=2, max_digits=4, min_operands=2, max_operands=3),
    }

    def sample(self, split_name: str, rng: random.Random) -> Example:
        spec = self.split_specs[split_name]
        count = rng.randint(spec.min_operands, spec.max_operands)
        numbers = [sample_int_with_digits(rng, spec.min_digits, spec.max_digits) for _ in range(count)]
        expr = " + ".join(str(x) for x in numbers)
        answer = str(sum(numbers))
        prompt = self._choose_template(split_name, rng).format(task=self.name, expr=expr)
        trace = {"columns": self._trace_columns(numbers)}
        metadata = {"numbers": numbers, "expr": expr, "dependencies": self.skill_dependencies()}
        return Example(prompt=prompt, answer=answer, task_name=self.name, split_name=split_name, metadata=metadata, trace=trace)

    @staticmethod
    def _trace_columns(numbers: Sequence[int]) -> List[Dict[str, int]]:
        rev_digits = [list(map(int, reversed(str(n)))) for n in numbers]
        max_len = max(len(d) for d in rev_digits)
        carry = 0
        columns = []
        for i in range(max_len):
            digits = [d[i] if i < len(d) else 0 for d in rev_digits]
            total = sum(digits) + carry
            out = total % 10
            carry_out = total // 10
            columns.append({"position": i, "digits": digits, "carry_in": carry, "digit_out": out, "carry_out": carry_out})
            carry = carry_out
        if carry:
            columns.append({"position": max_len, "digits": [], "carry_in": carry, "digit_out": carry, "carry_out": 0})
        return columns


class MultiplicationTask(BaseTask):
    name = "multiplication"
    primitive_skills = ("mul",)
    train_templates = (
        "Task: multiply the numbers.\nInput: {expr}\nAnswer:",
        "Compute the product.\n{expr} =",
    )
    ood_templates = (
        "Find the product of {expr}.\nReturn digits only.",
        "Multiply carefully: {expr}\nResult:",
    )
    split_specs = {
        "id": TaskSpec(min_digits=2, max_digits=3, min_operands=2, max_operands=2),
        "ood_length": TaskSpec(min_digits=4, max_digits=5, min_operands=2, max_operands=2),
        "ood_template": TaskSpec(min_digits=2, max_digits=3, min_operands=2, max_operands=2),
    }

    def sample(self, split_name: str, rng: random.Random) -> Example:
        spec = self.split_specs[split_name]
        a = sample_int_with_digits(rng, spec.min_digits, spec.max_digits)
        b = sample_int_with_digits(rng, spec.min_digits, spec.max_digits)
        expr = f"{a} * {b}"
        answer = str(a * b)
        prompt = self._choose_template(split_name, rng).format(task=self.name, expr=expr)
        metadata = {"operands": [a, b], "expr": expr, "dependencies": self.skill_dependencies()}
        trace = {"partial_products": self._partial_products(a, b)}
        return Example(prompt=prompt, answer=answer, task_name=self.name, split_name=split_name, metadata=metadata, trace=trace)

    @staticmethod
    def _partial_products(a: int, b: int) -> List[Dict[str, int]]:
        digits = list(map(int, reversed(str(b))))
        return [{"digit_index": i, "digit": d, "partial": a * d * (10 ** i)} for i, d in enumerate(digits)]


class LinearCombinationTask(BaseTask):
    name = "linear_combination"
    primitive_skills = ("add", "mul")
    train_templates = (
        "Task: evaluate the arithmetic expression.\nInput: {expr}\nAnswer:",
        "Compute the sum of products.\n{expr} =",
    )
    ood_templates = (
        "Evaluate exactly: {expr}\nDigits only.",
        "What does this equal?\n{expr}\nAnswer:",
    )
    split_specs = {
        "id": TaskSpec(min_digits=1, max_digits=2, min_operands=2, max_operands=3),
        "ood_length": TaskSpec(min_digits=2, max_digits=3, min_operands=4, max_operands=6),
        "ood_template": TaskSpec(min_digits=1, max_digits=2, min_operands=2, max_operands=3),
    }

    def sample(self, split_name: str, rng: random.Random) -> Example:
        spec = self.split_specs[split_name]
        term_count = rng.randint(spec.min_operands, spec.max_operands)
        pairs: List[Tuple[int, int]] = []
        terms: List[str] = []
        for _ in range(term_count):
            a = sample_int_with_digits(rng, spec.min_digits, spec.max_digits)
            b = sample_int_with_digits(rng, spec.min_digits, spec.max_digits)
            pairs.append((a, b))
            terms.append(f"{a}*{b}")
        expr = " + ".join(terms)
        answer_int = sum(a * b for a, b in pairs)
        prompt = self._choose_template(split_name, rng).format(task=self.name, expr=expr)
        trace = {"pairs": pairs, "partials": [a * b for a, b in pairs]}
        metadata = {"pairs": pairs, "expr": expr, "dependencies": self.skill_dependencies()}
        return Example(prompt=prompt, answer=str(answer_int), task_name=self.name, split_name=split_name, metadata=metadata, trace=trace)


class ExpressionTask(BaseTask):
    name = "expression"
    primitive_skills = ("add", "mul")
    train_templates = (
        "Task: evaluate the expression.\nInput: {expr}\nAnswer:",
        "Compute this expression.\n{expr} =",
    )
    ood_templates = (
        "Return only the numeric answer for:\n{expr}",
        "Evaluate exactly: {expr}\nAnswer:",
    )
    split_specs = {
        "id": TaskSpec(min_digits=1, max_digits=2, max_depth=2, min_operands=2, max_operands=2),
        "ood_length": TaskSpec(min_digits=2, max_digits=3, max_depth=3, min_operands=2, max_operands=2),
        "ood_template": TaskSpec(min_digits=1, max_digits=2, max_depth=2, min_operands=2, max_operands=2),
    }

    def sample(self, split_name: str, rng: random.Random) -> Example:
        spec = self.split_specs[split_name]
        expr, trace = self._gen_expr(rng, spec.min_digits, spec.max_digits, depth=spec.max_depth or 2)
        value = safe_eval_arithmetic(expr)
        prompt = self._choose_template(split_name, rng).format(task=self.name, expr=expr)
        metadata = {"expr": expr, "dependencies": self.skill_dependencies(), "depth": spec.max_depth}
        return Example(prompt=prompt, answer=str(value), task_name=self.name, split_name=split_name, metadata=metadata, trace=trace)

    def _gen_expr(self, rng: random.Random, min_digits: int, max_digits: int, depth: int) -> Tuple[str, Mapping[str, object]]:
        if depth <= 0:
            value = sample_int_with_digits(rng, min_digits, max_digits)
            return str(value), {"type": "leaf", "value": value}
        left_expr, left_trace = self._gen_expr(rng, min_digits, max_digits, depth - 1)
        right_expr, right_trace = self._gen_expr(rng, min_digits, max_digits, depth - 1)
        op = rng.choice(["+", "*"])
        expr = f"({left_expr} {op} {right_expr})"
        return expr, {"type": "node", "op": op, "left": left_trace, "right": right_trace}


class ValueAssignmentTask(BaseTask):
    name = "value_assignment"
    primitive_skills = ("mapping",)
    train_templates = (
        "Task: translate symbols.\nMap: {mapping}\nInput: {expr}\nAnswer:",
        "Use the mapping {mapping}\nTranslate: {expr}\nOutput:",
    )
    ood_templates = (
        "Apply this symbol table: {mapping}\nString: {expr}\nTranslated string:",
        "Translate with map [{mapping}].\nInput sequence: {expr}\nAnswer:",
    )
    split_specs = {
        "id": TaskSpec(alphabet_size=4, input_length=6),
        "ood_length": TaskSpec(alphabet_size=7, input_length=14),
        "ood_template": TaskSpec(alphabet_size=4, input_length=6),
    }

    def sample(self, split_name: str, rng: random.Random) -> Example:
        spec = self.split_specs[split_name]
        symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        outputs = list("0123456789")
        rng.shuffle(symbols)
        rng.shuffle(outputs)
        alpha_size = int(spec.alphabet_size or 4)
        mapping_pairs = list(zip(symbols[:alpha_size], outputs[:alpha_size]))
        mapping = dict(mapping_pairs)
        seq = "".join(rng.choice(symbols[:alpha_size]) for _ in range(int(spec.input_length or 6)))
        translated = "".join(mapping[ch] for ch in seq)
        mapping_str = ", ".join(f"{k}={v}" for k, v in mapping_pairs)
        prompt = self._choose_template(split_name, rng).format(mapping=mapping_str, expr=seq, task=self.name)
        metadata = {"mapping": mapping, "seq": seq, "dependencies": self.skill_dependencies()}
        trace = {"pairs": mapping_pairs}
        return Example(prompt=prompt, answer=translated, task_name=self.name, split_name=split_name, metadata=metadata, trace=trace)


class ECARolloutTask(BaseTask):
    name = "eca_rollout"
    primitive_skills = ("local_rule",)
    train_templates = (
        "Task: one-dimensional cellular automaton.\nRule: {rule}\nState: {expr}\nNext state:",
        "Apply elementary CA rule {rule} to {expr}\nOutput:",
    )
    ood_templates = (
        "Cellular automaton update.\nRule {rule}, current row {expr}\nReturn the next row only.",
        "Given ECA rule {rule}, evolve this row by one step:\n{expr}\nAnswer:",
    )
    split_specs = {
        "id": TaskSpec(state_length=9, steps=1),
        "ood_length": TaskSpec(state_length=19, steps=2),
        "ood_template": TaskSpec(state_length=9, steps=1),
    }

    def sample(self, split_name: str, rng: random.Random) -> Example:
        spec = self.split_specs[split_name]
        rule = rng.choice([30, 45, 90, 110, 184])
        state_length = int(spec.state_length or 9)
        state = "".join(rng.choice("01") for _ in range(state_length))
        steps = int(spec.steps or 1)
        states = [state]
        current = state
        for _ in range(steps):
            current = self._eca_step(current, rule)
            states.append(current)
        prompt = self._choose_template(split_name, rng).format(rule=rule, expr=state, task=self.name)
        metadata = {"rule": rule, "state": state, "steps": steps, "dependencies": self.skill_dependencies()}
        trace = {"states": states}
        return Example(prompt=prompt, answer=current, task_name=self.name, split_name=split_name, metadata=metadata, trace=trace)

    @staticmethod
    def _eca_step(state: str, rule: int) -> str:
        rule_bits = f"{rule:08b}"
        out = []
        extended = "0" + state + "0"
        for i in range(1, len(extended) - 1):
            neighborhood = extended[i - 1 : i + 2]
            idx = 7 - int(neighborhood, 2)
            out.append(rule_bits[idx])
        return "".join(out)
