# ArithCL Benchmark

ArithCL is a procedural benchmark for continual learning on arithmetic and related synthetic reasoning tasks.  
It is designed around four ideas:

1. **Sequential exposure:** tasks are learned one after another, not jointly.
2. **Controlled OOD testing:** each task has infinite generators for in-distribution and held-out splits.
3. **Positive transfer:** later tasks are explicitly compositional, so the benchmark can reward reuse of earlier skills.
4. **Sample efficiency:** the runner records adaptation curves and compares continual learning against a fresh-from-scratch baseline on each stage.

## Included task families

Core arithmetic:
- addition
- multiplication
- linear combination (sum of products)
- nested arithmetic expression evaluation

Optional synthetic warmups:
- value assignment
- elementary cellular automaton rollout

## Scenarios

### `three_stage_arith_v1`
Addition → Multiplication → Linear combination

### `extended_arith_v1`
Addition → Multiplication → Linear combination → Nested expressions

### `warmup_transfer_v1`
Value assignment → ECA rollout → Addition → Multiplication → Linear combination

## Evaluation splits

Each task includes:
- `id`: held-out samples from the training range
- `ood_length`: harder generalization to longer inputs / more terms / deeper trees
- `ood_template`: lexical or prompt-format shift with the same underlying computation

## Metrics

The package computes:
- final average accuracy
- forward transfer (FWT)
- final backward transfer (BWT)
- mean max forgetting
- zero-shot composition score
- examples-to-threshold and transfer-efficiency gain vs a fresh model

## Quick start

```bash
cd arithcl_benchmark
python -m arithcl_benchmark.cli --scenario extended_arith_v1 --baseline composable_oracle
```

## Wrapping your own learner

Your learner only needs two methods:

```python
class MyLearner:
    def update(self, batch):
        # batch is a list[Example]
        # use ex.prompt for the textual input and ex.answer as supervised target
        ...

    def predict(self, batch):
        # return list[str]
        return [...]
```

Then run:

```python
from arithcl_benchmark.runner import BenchmarkRunner
from arithcl_benchmark.scenarios import extended_arith_v1

runner = BenchmarkRunner(extended_arith_v1(), base_seed=0)
results = runner.run(MyLearner)
```

## Included sanity baselines

- `blank`: predicts nothing
- `stage_oracle`: solves tasks only after that exact task has been seen
- `composable_oracle`: solves any task whose primitive dependencies have already been learned

The oracle baselines are intentionally symbolic sanity checks. They are useful because they validate that the benchmark’s transfer metrics react differently when a learner can compose prior skills before direct supervision on a new task.
