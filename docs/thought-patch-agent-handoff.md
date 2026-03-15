# Agent Handoff: Fix Qwen Thought Patching and Re-run E2E

This document is an actionable handoff for another coding agent. The goal is to implement the remaining paper-grounded fix for Qwen thought patching, run the benchmark end to end, and produce a robust evaluation that separates real algorithmic gains from formatting artifacts.

Use this as an execution guide, not as background reading.

## Objective

Make the arithmetic thought-patching benchmark in `qwen_thought_patch_cli.py` as faithful as possible to:

- `learning-without-training.pdf`
- `transmuting-prompts-into-weights.pdf`

and then run a principled end-to-end benchmark on the local `Qwen3-0.6B` checkpoint at:

- `/content/drive/MyDrive/flair/software/qwen3/checkpoints/Qwen3-0.6B`

Success means:

1. The implementation matches the paper’s algorithmic structure for a gated pre-norm no-bias MLP block.
2. The benchmark reports clean paper-style metrics.
3. The evaluation is robust enough to distinguish:
   - arithmetic failure
   - formatting failure
   - implementation failure
   - model-capacity limitation

Do not claim `100%` unless the run actually achieves it.

## Files that matter

- `/content/drive/MyDrive/flair/software/continual-learning/nanoqwen/model.py`
- `/content/drive/MyDrive/flair/software/continual-learning/qwen_thought_patch_cli.py`
- `/content/drive/MyDrive/flair/software/continual-learning/tests/test_nanoqwen_thought_patches.py`
- `/content/drive/MyDrive/flair/software/continual-learning/tests/test_qwen_thought_patch_cli.py`
- `/content/drive/MyDrive/flair/software/continual-learning/docs/readable-log.md`
- `/content/drive/MyDrive/flair/software/continual-learning/docs/thought-patching-qwen.md`

## Current known state

Existing completed runs show:

- unfiltered multiply: `80%` prompted, `5%` raw, `10%` patched
- unfiltered sum: `95%` prompted, `0%` raw, `20%` patched
- filtered multiply sequential: `100%` prompted, `0%` raw, `50%` patched
- filtered multiply batch: `100%` prompted, `0%` raw, `70%` patched

The strongest run is:

- `/content/drive/MyDrive/flair/software/continual-learning/runs/qwen_thought_patch_multiply_filtered_batch.json`

The main known bottleneck is that the current Qwen implementation still relies on a synthetic output-side `d_bias`, while the paper-grounded no-bias adaptation should absorb the thought vector into the output projection.

## Paper-grounded algorithm you must implement

### 1. Exact theorem to respect

From `learning-without-training.pdf`, Theorem `B.2`:

- the context effect in a transformer block with skip connections is equivalent to:
  - a rank-1 first-layer weight update
  - an output-side vector update

The exact token patch is:

```text
delta_x = A(C, x) - A(C\I, x)
Delta_x = (W delta_x a_x^T) / ||a_x||^2
Delta b'_x = delta_x
```

This is token-dependent, but it tells you what the reusable approximation has to preserve.

### 2. Reusable thought patch to approximate

From `transmuting-prompts-into-weights.pdf`, Theorem `3.1` and Algorithm `1`:

At each layer:

1. Run a contextual pass on `[I, x]`.
2. Run a non-contextual pass on `[x]` using the current patches.
3. Compute `dz_l = A_l - a_l`.
4. Average `dz_l` into a thought vector.
5. Solve a least-squares update for the first MLP projection.
6. Apply patches layer by layer.
7. Recompute the non-contextual pass as patches accumulate.

### 3. Architectural adaptation to follow for Qwen

Qwen matches the important parts of the Gemma appendix:

- pre-MLP `RMSNorm`
- gated MLP with separate `gate` and `up` projections
- no explicit MLP output bias

So you should follow the adaptation pattern in `transmuting-prompts-into-weights.pdf`, Appendix `A`:

- use normalized MLP inputs for the input-side least-squares solve
- solve separate updates for the two gated input projections
- absorb the output-side vector into the down projection instead of inventing a new bias parameter

For Qwen, that means:

- `fc1` and `fc2` should remain separate first-layer solves
- `fc3` should become the primary mechanism for the output-side vector absorption
- the synthetic `d_bias` should not remain the main paper-path mechanism

## Implementation tasks

### Task 1: Make the no-bias output absorption first-class

Change the implementation so that the output-side shift is represented through `fc3` rather than a synthetic `d_bias`.

Concretely:

1. Inspect how `ThoughtPatch` is used in:
   - `nanoqwen/model.py`
   - `qwen_thought_patch_cli.py`
2. Keep `d_bias` only if needed for ablation or debugging.
3. Change the default benchmark path so the paper-grounded output mechanism is:
   - `d_fc1`
   - `d_fc2`
   - `d_fc3`
4. Do not let the “best” benchmark path depend on synthetic architecture changes.

### Task 2: Define the right `fc3` target

This is the most important technical choice.

Current model structure in `TransformerBlock.forward` is:

```text
att_resid = attention_residual_output
mlp_in = norm2(att_resid)
mlp_hidden = silu(fc1(mlp_in)) * fc2(mlp_in)
mlp_out = fc3(mlp_hidden)
resid_out = att_resid + mlp_out
```

For the contextual pass you already have:

- `att_resid_ctx`
- `mlp_in_ctx`
- `mlp_hidden_ctx`
- `mlp_out_ctx`
- `resid_out_ctx`

For the non-contextual pass you already have:

- `att_resid_raw`
- `mlp_in_raw`
- `mlp_hidden_raw`
- `mlp_out_raw`
- `resid_out_raw`

The theorem says the missing context contributes an output-side shift equal to `dz = att_resid_ctx - att_resid_raw`.

Because Qwen has no explicit output bias, the natural target for the patched MLP output is:

```text
target_mlp_out = mlp_out_ctx + dz
```

so the `fc3` solve should fit:

```text
(W3_patched + dW3) h_raw  ~=  target_mlp_out
```

which implies an update target like:

```text
target_fc3 = (mlp_out_ctx + dz) - mlp_out_raw
```

Use the current raw hidden state after the current patches have already affected `fc1` and `fc2`.

This is the most plausible Qwen analogue of the Gemma appendix, simplified by the fact that this Qwen implementation has no post-MLP RMSNorm.

### Task 3: Keep the input-side solves on normalized MLP inputs

Do not regress this part.

Use:

- `mlp_in` as the normalized activation
- separate least-squares solves for `fc1` and `fc2`

The current direction is already close:

```text
d_fc1: solve a -> W1_patched dz
d_fc2: solve a -> W2_patched dz
```

Retain the “current patched weights” behavior, because Algorithm `1` in the paper explicitly patches layer by layer and reruns the approximation pass under the current patch state.

### Task 4: Make the benchmark path use the paper-grounded default

Once the `fc3` path is implemented:

1. Make the benchmark default use the no-bias absorption path.
2. Keep ablation flags for:
   - `fc1/fc2` only
   - `fc1/fc2/fc3`
   - optional synthetic bias fallback if you want a debug mode
3. Make the final reported run use the paper-grounded path, not the debug path.

## Evaluation requirements

Do not rely on a single metric.

The benchmark must report at least these three views:

### 1. Paper-style task metric

This remains the primary number:

- greedy generation
- normalized final answer
- exact-match accuracy

This is what should populate the paper-style table:

- original with context
- original without context
- patched without context

### 2. Teacher-forced answer metric

Add a diagnostic metric over the answer tokens under teacher forcing.

At minimum report one of:

- exact next-token accuracy on answer tokens
- average log-prob of the gold answer tokens
- both if easy

Why:

- if generation exact-match is poor but teacher-forced answer quality is high, the algorithm may be right and the remaining issue is output format
- if both are poor, the patch itself is poor

### 3. Formatting-aware diagnostic metric

Add one softer generation diagnostic, such as:

- whether the expected integer appears anywhere in the generated string
- whether the normalized final integer matches

This lets you separate:

- “model reasoned correctly but formatted badly”
- “model never reached the right answer”

## Benchmark protocol

### Phase 1: Sanity check

Run filtered multiplication first.

Command shape:

```bash
python3 qwen_thought_patch_cli.py \
  --model-size 0.6B \
  --model-type instruct \
  --device cuda \
  --local-dir /content/drive/MyDrive/flair/software/qwen3/checkpoints/Qwen3-0.6B \
  --tasks multiply \
  --train-examples 10 \
  --eval-examples 20 \
  --seeds 1 \
  --fit-mode layer_batch \
  --max-new-tokens 32 \
  --no-step-eval \
  --out runs/qwen_thought_patch_multiply_filtered_fc3.json
```

Use the filtered dataset regime first, because it reproduces the paper’s intended baseline:

- with context correct
- without context incorrect

Before trusting any patch result, confirm the benchmark shows:

- contextual eval: `100%`
- raw eval: near `0%`, ideally `0%`

If not, stop and fix the prompt or the filtering before changing the algorithm.

### Phase 2: Ablations

Run at least these ablations:

1. `fc1/fc2` only
2. `fc1/fc2/fc3`
3. optional debug run with synthetic `d_bias`

This will tell you whether `fc3` is actually carrying the missing paper-grounded behavior.

### Phase 3: Robust evaluation

For the best configuration, run:

- multiply, filtered, `5` seeds if runtime allows
- sum, filtered, `5` seeds if runtime allows

If runtime is too high, do:

- `1` seed for development
- `5` seeds only after the implementation stabilizes

### Phase 4: Unfiltered regression check

After the filtered benchmark improves, rerun:

- multiply unfiltered
- sum unfiltered

This checks whether the fix only works in the tightly controlled paper regime or actually improves the broader benchmark path too.

## What to log

For each run, log:

- model, checkpoint path, device, dtype
- task, seed, train size, eval size
- baseline contextual accuracy
- baseline raw accuracy
- patched eval accuracy
- teacher-forced answer metric
- formatting-aware metric
- patch norms:
  - `||d_fc1||`
  - `||d_fc2||`
  - `||d_fc3||`
  - optional `||d_bias||` if kept for debug
- runtime
- a few representative wrong examples

The CLI output should make it easy to answer:

1. Did the patch improve arithmetic behavior?
2. Did `fc3` actually activate?
3. Is failure mostly arithmetic or mostly formatting?

## Troubleshooting checklist

### If contextual baseline is below 100% on filtered multiply

Problem:

- prompt template or filtering is wrong
- do not touch the patch algorithm yet

Check:

1. The contextual prompt format.
2. The raw prompt format.
3. The filtering condition:
   - prompted correct
   - raw incorrect
4. Whether answer tokens are included in both contextual and non-contextual teacher-forced passes.

### If raw baseline is not near 0% on filtered multiply

Problem:

- the filter is not enforcing the paper-style setup

Check:

1. The filtering loop in `make_examples`.
2. Whether normalization of generated answers is too permissive.
3. Whether the raw query prompt accidentally contains instruction-like cues.

### If `fc3` norm stays zero or tiny after the fix

Problem:

- the output-absorption path is not actually wired in
- or its target is degenerate

Check:

1. Whether `patch_fc3` is enabled on the main path.
2. Whether the `fc3` least-squares solve is executed.
3. Whether the target uses:
   - contextual `mlp_out`
   - `dz`
   - raw `mlp_hidden`
4. Whether dtype conversions are accidentally zeroing the update.

### If patch norms explode

Problem:

- unstable least-squares solve
- target mismatch
- learning rate too high

Check:

1. Add or increase ridge regularization.
2. Reduce learning rate.
3. Inspect per-layer norms.
4. Inspect a small number of token positions directly.
5. Compare sequential vs batch mode.

### If teacher-forced answer quality is good but generation accuracy is bad

Problem:

- formatting is the bottleneck, not arithmetic

Check:

1. Whether the model is producing explanations before the answer.
2. Whether `max_new_tokens` is too short.
3. Whether normalization should extract:
   - last integer
   - integer anywhere
4. Whether the instruction wording better enforces a one-integer answer.

Do not hide this with a looser metric. Report it explicitly.

### If both teacher-forced and generation metrics are poor

Problem:

- the patch is not reproducing the contextual computation

Check:

1. Alignment positions for query and answer tokens.
2. Whether the contextual and non-contextual prompts tokenize as expected.
3. Whether the non-contextual pass is recomputed under current patches.
4. Whether the layer order matches Algorithm `1`.
5. Whether the target uses the current patched weights for `fc1` and `fc2`.
6. Whether `fc3` target reflects the output-side thought vector absorption.

## How to search the papers and code quickly

Use fast local search. Do not guess.

### Search the papers

Use:

```bash
pdftotext learning-without-training.pdf - | rg -n "Theorem B.2|rank-1|bias|skip"
pdftotext transmuting-prompts-into-weights.pdf - | rg -n "Theorem 3.1|Algorithm 1|Gemma|Wdown|least-squares|Table 1"
```

The most relevant sections are:

- `learning-without-training.pdf`
  - Theorem `B.2`
  - the stack-of-blocks discussion after it
- `transmuting-prompts-into-weights.pdf`
  - Theorem `3.1`
  - Algorithm `1`
  - Appendix `A` on Gemma adaptation
  - Appendix `B.1` on arithmetic experiments

### Search the code

Use:

```bash
rg -n "ThoughtPatch|d_bias|patch_fc3|fit_layer_batch|fit_one_example|mlp_out|att_resid|build_alignment_positions" nanoqwen/model.py qwen_thought_patch_cli.py
```

The main places to inspect are:

- where traces are captured
- where layer targets are defined
- where `fc3` is solved
- where evaluation normalization is defined

### Use short one-off probes

When unsure, run targeted probes instead of full benchmarks. For example:

- print tokenized prompt ids for contextual vs raw prompts
- print aligned query and answer positions
- print one layer’s:
  - `||dz||`
  - `||d_fc1||`
  - `||d_fc2||`
  - `||d_fc3||`
- print a few wrong generations

This is faster than rerunning the full benchmark blindly.

## Required tests before the expensive benchmark

Before running the full end-to-end benchmark:

```bash
python3 -m py_compile \
  nanoqwen/model.py \
  qwen_thought_patch_cli.py \
  tests/test_nanoqwen_thought_patches.py \
  tests/test_qwen_thought_patch_cli.py

pytest -q tests/test_nanoqwen_thought_patches.py tests/test_qwen_thought_patch_cli.py
```

Add or update tests for:

1. `fc3` patch application changes logits.
2. Zero patch remains identity.
3. The new `fc3` target path is exercised.
4. Teacher-forced metrics compute without generation.

## Definition of done

The task is complete when all of the following are true:

1. The implementation uses a paper-grounded no-bias output absorption path for Qwen.
2. The best benchmark path no longer depends on synthetic `d_bias`.
3. The CLI reports:
   - paper-style exact-match accuracy
   - teacher-forced answer metric
   - formatting-aware diagnostic metric
4. Filtered multiply is rerun end to end and the result is saved in `runs/`.
5. The final doc update states clearly:
   - exact numbers
   - whether the fix improved over `70%`
   - whether any remaining gap looks like implementation mismatch or model-capacity limitation

## Final reporting format

When you finish, update:

- `/content/drive/MyDrive/flair/software/continual-learning/docs/readable-log.md`
- `/content/drive/MyDrive/flair/software/continual-learning/docs/thought-patching-qwen.md`

and include:

1. the exact commands run
2. the exact output files written
3. the best filtered multiply result
4. whether `fc3` materially improved the benchmark
5. the remaining bottleneck if parity is still not reached

The correct outcome is not “claim success.” The correct outcome is “make the implementation more faithful to the papers, run the benchmark honestly, and explain the remaining gap clearly.”
