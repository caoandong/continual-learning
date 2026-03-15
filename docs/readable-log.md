# Qwen Thought Patching: Current Progress and Bottlenecks

This note summarizes the current state of the Qwen `0.6B` thought-patching implementation, the strongest completed end-to-end results, and the main gaps between the current code and the algorithms described in:

- `learning-without-training.pdf`
- `transmuting-prompts-into-weights.pdf`

It is intended as a readable execution log, not a paper recap.

## Current status

The end-to-end benchmark path is implemented and runnable through:

- `qwen_thought_patch_cli.py`
- `nanoqwen/model.py`

Completed artifacts already exist in `runs/`:

- `runs/qwen_thought_patch_smoke.json`
- `runs/qwen_thought_patch_multiply_10_20.json`
- `runs/qwen_thought_patch_sum_10_20.json`
- `runs/qwen_thought_patch_multiply_filtered_seq.json`
- `runs/qwen_thought_patch_multiply_filtered_batch.json`

So the answer to "did the e2e test work?" is:

- Yes, the CLI runs end to end.
- No, it does not yet reproduce the paper's `100%` patched arithmetic result on local `Qwen3-0.6B`.

## Best completed benchmark results

### 1. Unfiltered Table-1-style arithmetic

From the completed local runs:

| Run | Original w/ context | Original w/o context | Patched w/o context |
| --- | ---: | ---: | ---: |
| `runs/qwen_thought_patch_multiply_10_20.json` | `80%` | `5%` | `10%` |
| `runs/qwen_thought_patch_sum_10_20.json` | `95%` | `0%` | `20%` |

These runs show that the basic pipeline works, but the raw baseline was not fully controlled because the instructed prompt was not perfect on this model/checkpoint.

### 2. Filtered multiplication benchmark

The most informative run is the filtered multiplication setup, where examples are selected so that:

- the prompted model is correct
- the raw model is incorrect

This gives a cleaner paper-style sanity check on `Qwen3-0.6B`.

Results:

| Run | Original w/ context | Original w/o context | Patched w/o context |
| --- | ---: | ---: | ---: |
| `runs/qwen_thought_patch_multiply_filtered_seq.json` | `100%` | `0%` | `50%` |
| `runs/qwen_thought_patch_multiply_filtered_batch.json` | `100%` | `0%` | `70%` |

This is the strongest current result:

- `100%` contextual baseline
- `0%` raw baseline
- `70%` patched evaluation accuracy

That means the remaining problem is not prompt construction anymore. The bottleneck is the patch formulation and how it is applied in Qwen.

## What the papers say the core algorithm is

### 1. Exact token patch: `learning-without-training.pdf`

The core exact result is the skip-connection theorem for a transformer block with residual structure: Theorem `B.2`.

For a contextual block

```text
T(C, x) = A(C, x) + W' g(W A(C, x) + b) + b'
```

removing a context chunk can be made exactly equivalent to patching the block with:

```text
delta_x = A(C, x) - A(C\\I, x)
Delta_x = (W delta_x a_x^T) / ||a_x||^2
```

plus an output-side vector update:

```text
Delta b'_x = delta_x
```

where `a_x = A(C\I, x)`.

Interpretation:

- the instruction induces a rank-1 update to the first MLP matrix
- with skip connections, it also induces a bias-like output shift

This theorem is exact, but token-dependent.

### 2. Reusable thought patch: `transmuting-prompts-into-weights.pdf`

The reusable version is Theorem `3.1` plus Algorithm `1`.

The paper replaces token-specific patches with a single token-independent thought patch:

```text
thought vector:
  delta(I) = mean_i delta_i

thought matrix:
  Delta(I) = argmin_M sum_i ||M a_i - W delta_i||^2
```

Operationally, Algorithm `1` says:

1. Run a contextual pass on `[I, x]`.
2. Run a non-contextual pass on `[x]` using the current patches.
3. At each layer, compute `dz_l = A_l - a_l`.
4. Set the thought vector update to the mean discrepancy.
5. Solve a least-squares update for the first MLP projection.
6. Apply the update layer by layer, rerunning the non-contextual pass as patches accumulate.

That is the core algorithm the implementation should follow.

## What is currently correct in the implementation

The current code already matches several important parts of the papers:

- Layer-wise trace capture is implemented in `nanoqwen/model.py`.
- The CLI does contextual and non-contextual passes with teacher forcing.
- Query tokens and answer tokens are aligned across both passes.
- The filtered benchmark can produce the paper-style regime where:
  - contextual accuracy is `100%`
  - raw accuracy is `0%`
- The batch least-squares mode is materially better than naive sequential fitting:
  - `70%` vs `50%` on the filtered multiplication benchmark.

So the implementation is not fundamentally broken. It is partially right and already captures a real fraction of the effect.

## Key bottlenecks and issues

### 1. The output-side update is not grounded in the exact theorem

This is the main research-to-code mismatch.

`learning-without-training.pdf` Theorem `B.2` is explicit: for transformer blocks with skip connections, the patch is not only a first-layer matrix update. It also requires an output-side vector shift.

The current Qwen code models that shift with a synthetic `d_bias` attached after `fc3` in `nanoqwen/model.py`.

Why this is a problem:

- Qwen's MLP does not have a native output bias.
- The current implementation effectively changes the architecture to add one.
- `transmuting-prompts-into-weights.pdf` Appendix `A` does not do that for Gemma. Instead, it absorbs the vector update into the output projection with an additional least-squares solve over `W_down`.

Practical consequence:

- the strongest current run, `runs/qwen_thought_patch_multiply_filtered_batch.json`, still has `fc3_norm = 0.0`
- this means the best current result was achieved without using the output projection update at all
- that is not the paper-grounded solution for a no-bias gated MLP block

This is the clearest bottleneck.

### 2. Qwen needs the Gemma-style architectural adaptation, not the vanilla bias update

`transmuting-prompts-into-weights.pdf` Appendix `A` describes how to adapt the algorithm to gated pre-norm MLP blocks:

- use the normalized MLP input for the first-layer least-squares solve
- solve separate updates for the two gated input projections
- absorb the output-side vector through the down projection when no explicit bias exists

Qwen has the same relevant structural properties:

- pre-MLP `RMSNorm`
- gated MLP (`fc1` and `fc2`)
- no explicit MLP output bias

So the Qwen adaptation should follow the same pattern:

- solve separate least-squares updates for `fc1` and `fc2`
- solve an additional least-squares update for `fc3` that absorbs the output-side thought vector

The current code does the first part well enough, but the second part is still incomplete.

### 3. The strongest failures are answer-format failures, not only arithmetic failures

The wrong predictions in `runs/qwen_thought_patch_multiply_filtered_batch.json` are not always pure arithmetic mistakes.

Representative failures look like:

- expected `135`, predicted normalized `5`
- expected `126`, predicted normalized `1`
- expected `225`, predicted normalized `22`

The model often starts a correct verbal derivation but truncates before the final integer is fully emitted. This shows a second bottleneck:

- the patch partially transfers the task
- but it does not reliably transfer the paper's short-answer response style

This is consistent with `transmuting-prompts-into-weights.pdf`, which notes that the least-squares formulation can overfit to specific completion surfaces. In the current Qwen setup, the patch appears to encode "discuss multiplication" more easily than "emit exactly one short integer token sequence."

### 4. The current benchmark is generation-limited

The evaluation is greedy generation on the patched model with a normalizer that extracts the last integer from the decoded response.

That is reasonable, but it is brittle when the model produces:

- partial derivations
- explanatory prose
- truncated numeric answers

This does not invalidate the result, but it does mean that patched accuracy is sensitive to response formatting. The paper's arithmetic result is easiest to reproduce when the model is already very compliant with the short-answer template.

### 5. `Qwen3-0.6B` is probably below the paper's capacity regime

The paper's arithmetic table is reported on `Gemma 3 1B Instruction Tuned`.

The local benchmark uses `Qwen3-0.6B`, which is:

- smaller
- architecturally different
- weaker on this exact arithmetic instruction-following behavior

This shows up clearly in the unfiltered runs, where the instructed baseline itself was only `80%` to `95%`.

Even after forcing the cleaner filtered setting, the patched result still tops out at `70%`, not `100%`.

So there are two separate bottlenecks:

- implementation mismatch
- model/checkpoint capacity gap

Both matter.

### 6. The batch implementation is slow

The batch benchmark reaches the best current accuracy, but it is expensive:

- `runs/qwen_thought_patch_multiply_filtered_batch.json` fit time: about `149.5s`
- `runs/qwen_thought_patch_multiply_filtered_seq.json` fit time: about `31.3s`

The reason is straightforward:

- the current batch mode repeatedly reruns non-contextual traces across layers and examples
- the implementation is faithful to the iterative patching logic, but not efficient

This is a runtime bottleneck, not the main accuracy bottleneck.

## Bottom line

Current conclusion:

- the e2e CLI works
- the local `Qwen3-0.6B` setup already reaches a real patching effect
- the best completed benchmark is `70%` patched eval on a filtered multiplication benchmark with `100%` contextual and `0%` raw baselines
- the main blocker to paper parity is that the Qwen implementation still uses a synthetic output bias instead of the paper-grounded output-projection absorption step for no-bias gated MLPs

## Most important next fix

The next implementation step should be:

1. Remove the synthetic dependence on `d_bias` as the primary output-side mechanism.
2. Make the `fc3` solve first-class for Qwen.
3. Use the paper-grounded no-bias adaptation:
   - first-layer solves on normalized `mlp_in`
   - separate solves for `fc1` and `fc2`
   - output projection solve on `fc3` to absorb the thought vector into `down_proj`
4. Re-run the filtered multiplication benchmark first.

If that still does not reach parity, the remaining gap is likely model capacity rather than algorithmic misunderstanding.
