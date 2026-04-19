# ArithCL Benchmark Specification

## Goal

ArithCL is an explicit benchmark for continual learning under **sequential exposure**, **OOD rule generalization**, **retention**, and **positive transfer**.

The benchmark is built for the setting you described:
- task 1 teaches a primitive arithmetic operator,
- task 2 teaches a second primitive,
- task 3 requires recombining the previous primitives,
- optional later tasks deepen composition.

## Why these design choices

### 1. Procedural generation instead of fixed datasets
Every task is generated on demand. This avoids contamination from static test sets and makes it possible to measure OOD generalization cleanly.

### 2. Arithmetic as a microscope, not the whole story
Arithmetic tasks are ideal for studying rule learning because they have exact answers, controllable length/value difficulty, and known algorithmic decompositions. But they are still cleaner than natural language, so the benchmark includes optional non-arithmetic warmups.

### 3. Composition is a first-class metric
The benchmark is not limited to “did the model forget?”  
It asks: **does earlier learning reduce the sample complexity of later learning?**

That is why the core scenarios place `linear_combination` and `expression` *after* addition and multiplication.

## Task families

### Primitive tasks
- `addition`: multi-operand integer addition
- `multiplication`: integer multiplication

### Compositional tasks
- `linear_combination`: sum of products such as `12*7 + 5*19 + 3*8`
- `expression`: nested arithmetic expressions using `+` and `*`

### Optional warmup tasks
- `value_assignment`: symbolic translation with a provided mapping
- `eca_rollout`: elementary cellular automaton rollout

These warmup tasks are included because synthetic non-arithmetic structure may be useful for studying transfer without leaking the downstream arithmetic tasks.

## Splits

Each task supports three evaluation splits.

### `id`
Held-out examples from the training regime.

### `ood_length`
Harder examples that increase at least one structural axis:
- more digits
- more operands / terms
- deeper expression trees
- longer symbolic sequences

### `ood_template`
Different prompt formats with the same underlying computation.

## Metrics

### Final average accuracy
Mean final accuracy across tasks on each split.

### Forward transfer
For task `t_j`, evaluate performance **before** training on `t_j`, after learning tasks `1..j-1`, and subtract the initial baseline:
`FWT_j = A(before j, j) - A(init, j)`

### Final backward transfer
For task `t_j`, compare final performance to performance immediately after learning that task:
`BWT_j = A(final, j) - A(after learning j, j)`

### Mean max forgetting
For task `t_j`, compare the best post-learning score seen during the run to the final score.

### Zero-shot composition
Average pre-training accuracy on tasks with more than one primitive dependency, measured before their own stage.

### Examples-to-threshold
For each stage, measure how many examples are needed to hit a fixed target on a chosen split.  
The benchmark compares:
- continual learner on the stage
- fresh learner trained from scratch on only that stage

This yields a **transfer-efficiency gain**.

## Recommended reporting

For any learner, report:
- per-task adaptation curves
- task-by-time accuracy matrix
- FWT / BWT / forgetting
- zero-shot composition
- examples-to-threshold on `ood_length`

## Scenarios

### `three_stage_arith_v1`
1. addition
2. multiplication
3. linear combination

This is the smallest clean scenario matching the proposed research question.

### `extended_arith_v1`
1. addition
2. multiplication
3. linear combination
4. expression

Adds a second composition level.

### `warmup_transfer_v1`
1. value assignment
2. ECA rollout
3. addition
4. multiplication
5. linear combination

This is the track for testing whether synthetic non-arithmetic pre-curricula improve later continual arithmetic learning.

## Included sanity baselines

The package includes symbolic baselines to verify that the benchmark behaves as intended.

### `stage_oracle`
Learns only after the stage is directly seen.  
Expected behavior:
- no forgetting
- little or no zero-shot composition
- no positive forward transfer on compositional tasks

### `composable_oracle`
Acquires primitive skills and solves any future task whose dependencies are already satisfied.  
Expected behavior:
- no forgetting
- strong zero-shot composition
- positive forward transfer on `linear_combination` and `expression`

If your metric summary fails to separate these two baselines, the benchmark implementation is wrong.
