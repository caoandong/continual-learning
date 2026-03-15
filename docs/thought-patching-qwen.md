# Thought Patching for Qwen in `nanoqwen/model.py`

This note reviews:

- `transmuting-prompts-into-weights.pdf`
- `learning-without-training.pdf`

and turns the core algorithm into a concrete implementation plan for the Qwen model in [`nanoqwen/model.py`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py).

## Runnable CLI

The repository now includes a standalone runnable script:

- [`qwen_thought_patch_cli.py`](/Volumes/SB-XTM5/flair/software/continual-learning/qwen_thought_patch_cli.py)

It does the full end-to-end loop:

- loads a Qwen checkpoint and tokenizer
- builds the arithmetic train/eval benchmark in memory
- fits the thought patch with the layer-wise least-squares procedure
- evaluates the prompted, raw, and patched model
- prints readable step logs and final summary tables
- writes a JSON artifact with all metrics

The CLI uses the local 0.6B checkpoint by default when it exists at:

- `/Volumes/SB-XTM5/flair/software/qwen3/checkpoints/Qwen3-0.6B`

Recommended first run:

```bash
uv run python qwen_thought_patch_cli.py \
  --model-size 0.6B \
  --model-type instruct \
  --tasks multiply,sum \
  --train-examples 10 \
  --eval-examples 20 \
  --seeds 1 \
  --learning-rate 0.1 \
  --rho 0.0 \
  --max-new-tokens 32 \
  --out runs/qwen_thought_patch_metrics.json
```

For a paper-style average over multiple random seeds:

```bash
uv run python qwen_thought_patch_cli.py \
  --model-size 0.6B \
  --model-type instruct \
  --tasks multiply,sum \
  --train-examples 10 \
  --eval-examples 20 \
  --seeds 5 \
  --learning-rate 0.1 \
  --rho 0.0 \
  --max-new-tokens 32 \
  --out runs/qwen_thought_patch_metrics.json
```

The CLI prints three kinds of output:

- step-by-step patch-fit logs with per-step accuracy and patch norms
- a per-seed summary table
- a final paper-style table with:
  - original model with context
  - original model without context
  - patched model without context

If the contextual baseline is weak on `0.6B`, move to `1.7B`.

## Current implementation status

The implementation is now native to the Qwen codepath:

- [`nanoqwen/model.py`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py) supports:
  - `ThoughtPatch`
  - `build_empty_thought_patches(model)`
  - `model(..., thought_patches=patches, return_trace=True)`
  - patched greedy generation via `generate_text_simple(..., thought_patches=...)`
- [`qwen_thought_patch_cli.py`](/Volumes/SB-XTM5/flair/software/continual-learning/qwen_thought_patch_cli.py) uses those native APIs directly

So the document is no longer just a plan: the tracing, patch fitting, runtime patch application, CLI logging, and evaluation path are implemented.

## Observed results on local `Qwen3-0.6B`

Using the local checkpoint at `/Volumes/SB-XTM5/flair/software/qwen3/checkpoints/Qwen3-0.6B`, the current implementation behaves as follows on the Table 1-style arithmetic setup with `10` training examples and `20` held-out evaluation examples:

```text
Multiply numbers:
  original model w/ context    80.00%
  original model w/o context    5.00%
  patched model w/o context    10.00%

Sum numbers:
  original model w/ context    95.00%
  original model w/o context    0.00%
  patched model w/o context    20.00%
```

Interpretation:

- the local `0.6B` checkpoint follows the instructed arithmetic prompt reasonably well
- the raw query without instruction is poor, as expected
- the implemented thought patch improves the raw model, but does not reproduce the paper's `100%` patched Table 1 result on this checkpoint

That means the implementation path is working, but this particular model/checkpoint is not strong enough to match the Gemma-based result reported in the paper.

## 1. Start with the Table 1 sanity check

Use the smallest task first: multiplication of three single-digit numbers.

- Instruction `I`: `Multiply numbers.`
- Query `x`: `3, 4, 7 ->`
- Target answer: `84`

The important setup in this implementation is:

- Contextual prompt for the original model:
  `Instruction: multiply the numbers. Answer with one integer only. Query: 3, 4, 7.`
- Context-less prompt for the patched model:
  `Query: 3, 4, 7.`

During patch fitting, both passes must be teacher-forced on the same answer tokens so that layer activations are aligned:

```text
Contextual pass:
  user: "Instruction: multiply the numbers. Answer with one integer only. Query: 3, 4, 7."
  assistant: "84"

Non-contextual pass:
  user: "Query: 3, 4, 7."
  assistant: "84"
```

After fitting, inference uses only:

```text
Query: 3, 4, 7.
```

and the patched model should generate `84`.

Recommended first experiment:

- Model: Qwen instruct checkpoint, not base.
- Train patch on 10 random multiplication examples.
- Evaluate on 20 held-out random multiplication examples.
- Decode greedily.
- Compare:
  - original model with instruction
  - original model without instruction
  - patched model without instruction

The target behavior for the sanity check is still the same pattern as Table 1:

- with instruction: near-perfect
- without instruction and without patch: poor
- without instruction but with patch: near-perfect

## 2. What the two papers contribute

### `learning-without-training.pdf`

This paper gives the exact per-token theorem.

For one transformer block, removing a context chunk `I` can be replaced exactly by a token-dependent patch:

- a vector update
- a rank-1 update to the first MLP weight

For a pre-LN transformer block with residual connection, the exact token patch is:

```text
delta_i = A([I, x], x_i) - A([x], x_i)
Delta_i = (W delta_i a_i^T) / ||a_i||^2
```

where:

- `A(...)` is the contextual block output before the MLP
- `a_i = A([x], x_i)` is the same quantity without the instruction
- `W` is the first MLP matrix

With skip connections, the theorem also needs an additive output-side shift:

```text
b'_i = delta_i
```

So the exact result is:

- patch the first MLP projection with a rank-1 matrix
- patch the block output with a bias-like vector

The limitation is that this is query-dependent and token-dependent. It is exact, but not reusable.

### `transmuting-prompts-into-weights.pdf`

This paper turns the exact token patches above into reusable, token-independent thought patches.

For a dataset of completions consistent with instruction `I`, it defines:

```text
thought vector:
  delta(I) = mean_i delta_i

thought matrix:
  Delta(I) = argmin_M sum_i ||M a_i - W delta_i||^2
```

Closed form, when `Z = sum_i a_i a_i^T` is invertible:

```text
Delta(I) = (sum_i W delta_i a_i^T) Z^{-1}
```

Practical version:

- estimate the thought vector by averaging activation differences
- estimate the thought matrix with a least-squares or ridge solve
- apply the patch layer by layer
- rerun the non-contextual forward pass after each layer update

The paper's Algorithm 1 is the key operational recipe.

## 3. What this means for Qwen

Qwen in [`nanoqwen/model.py`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py) is not a vanilla MLP block.

Relevant structure:

- [`FeedForward`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py#L76) has:
  - `fc1`: gate projection
  - `fc2`: up projection
  - `fc3`: down projection
- [`TransformerBlock.forward`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py#L189) is:
  - attention residual
  - `norm2`
  - SwiGLU MLP
  - residual add

So Qwen matches the same high-level case as the Gemma appendix in the paper:

- pre-MLP RMSNorm
- gated MLP
- no explicit MLP output bias

But it is slightly simpler than Gemma because this implementation has no extra post-MLP RMSNorm.

That gives the following mapping for layer `l`.

### Per-layer quantities to capture

For the contextual pass on `[I, x]`, capture:

- `r_ctx_l`: residual stream after attention add, before `norm2`
- `a_ctx_l = norm2(r_ctx_l)`: normalized MLP input
- `h_ctx_l = silu(fc1(a_ctx_l)) * fc2(a_ctx_l)`: MLP hidden state
- `d_ctx_l = fc3(h_ctx_l)`: MLP output before residual add

For the non-contextual patched pass on `[x]`, capture:

- `r_l`
- `a_l = norm2(r_l)`
- `h_l`
- `d_l`

Then define:

```text
dz_l = r_ctx_l - r_l
db_l = mean(dz_l, over aligned answer tokens in the batch)
```

`dz_l` is the paper's per-token contextual discrepancy, and `db_l` is the Qwen version of the thought vector for that layer.

In the implemented CLI, the least-squares fit uses aligned positions for:

- the query tokens themselves
- the teacher-forced answer tokens

This matters. Fitting only on answer-token positions was too weak because the patch also needs to change how the model interprets the raw query before generation starts.

## 4. Concrete Qwen algorithm

For Qwen, patch all three MLP matrices.

### 4.1 Patch the two first-layer projections

Because the MLP is gated, solve two separate least-squares problems, one for each first-layer linear:

```text
dW_fc1_l = argmin_M ||a_l @ M^T - (dz_l @ W_fc1^T)||_F^2 + rho ||M||_F^2
dW_fc2_l = argmin_M ||a_l @ M^T - (dz_l @ W_fc2^T)||_F^2 + rho ||M||_F^2
```

This is the Qwen/SwiGLU analogue of the paper's single thought matrix.

### 4.2 Handle the missing output bias

The theory wants an additive output-side vector `db_l`, but Qwen's `fc3` has `bias=False`.

There are two ways to implement this.

Recommended:

- add a runtime-only patch bias tensor after `fc3`
- do not change the checkpoint weights on disk

In that case:

```text
bias_patch_l += eta * db_l
```

and patch `fc3` only for the remaining output mismatch:

```text
dW_fc3_l = argmin_M ||h_l @ M^T - (d_ctx_l - d_l)||_F^2 + rho ||M||_F^2
```

If you do not want to introduce a new bias term, absorb the thought vector into `fc3`:

```text
dgoal_l = (d_ctx_l - d_l) + db_l
dW_fc3_l = argmin_M ||h_l @ M^T - dgoal_l||_F^2 + rho ||M||_F^2
```

This is the simpler bias-free adaptation for this codebase.

### 4.3 Apply updates layer by layer

For each batch of training examples:

1. Run one contextual pass on `[I, x]` and cache all target traces.
2. For `l = 0..L-1`:
   - rerun the non-contextual pass on `[x]` using the current patches
   - compute `dz_l`, `db_l`, `dW_fc1_l`, `dW_fc2_l`, and `dW_fc3_l`
   - update the current layer patch with learning rate `eta`

This rerun is important. It matches Algorithm 1 in the paper: later layers should see the effect of earlier patched layers.

## 5. Minimal implementation sketch

This is the cleanest minimal version for this repo.

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class LayerThoughtPatch:
    d_fc1: torch.Tensor
    d_fc2: torch.Tensor
    d_fc3: torch.Tensor
    d_bias: torch.Tensor | None = None


def ridge_weight_update(src: torch.Tensor, target: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Solve min_D ||src @ D.T - target||_F^2 + rho ||D||_F^2
    src:    [n_tokens, d_in]
    target: [n_tokens, d_out]
    returns D with shape [d_out, d_in]
    """
    src32 = src.float()
    tgt32 = target.float()
    if rho > 0:
        gram = src32.T @ src32
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        rhs = src32.T @ tgt32
        return torch.linalg.solve(gram + rho * eye, rhs).T
    return torch.linalg.lstsq(src32, tgt32).solution.T


def fit_qwen_thought_patch(model, ctx_batch, raw_batch, answer_mask, rho=0.0, eta=1.0):
    """
    ctx_batch: token ids for prompts with instruction I
    raw_batch: token ids for prompts without I but with the same teacher-forced answers
    answer_mask: boolean mask selecting only answer tokens
    """
    n_layers = len(model.trf_blocks)
    patches = []
    for block in model.trf_blocks:
        ff = block.ff
        patches.append(
            LayerThoughtPatch(
                d_fc1=torch.zeros_like(ff.fc1.weight),
                d_fc2=torch.zeros_like(ff.fc2.weight),
                d_fc3=torch.zeros_like(ff.fc3.weight),
                d_bias=torch.zeros(ff.fc3.weight.size(0), device=ff.fc3.weight.device, dtype=ff.fc3.weight.dtype),
            )
        )

    ctx_trace = trace_qwen(model, ctx_batch, patches=None)

    for l, block in enumerate(model.trf_blocks):
        raw_trace = trace_qwen(model, raw_batch, patches=patches)

        dz = (ctx_trace[l]["att_resid"] - raw_trace[l]["att_resid"])[answer_mask]
        a = raw_trace[l]["mlp_in"][answer_mask]
        h = raw_trace[l]["mlp_hidden"][answer_mask]
        d = raw_trace[l]["mlp_out"][answer_mask]
        d_ctx = ctx_trace[l]["mlp_out"][answer_mask]

        db = dz.mean(dim=0)

        target_fc1 = dz @ block.ff.fc1.weight.T
        target_fc2 = dz @ block.ff.fc2.weight.T
        target_fc3 = d_ctx - d

        patches[l].d_fc1 += eta * ridge_weight_update(a, target_fc1, rho).to(block.ff.fc1.weight.dtype)
        patches[l].d_fc2 += eta * ridge_weight_update(a, target_fc2, rho).to(block.ff.fc2.weight.dtype)
        patches[l].d_fc3 += eta * ridge_weight_update(h, target_fc3, rho).to(block.ff.fc3.weight.dtype)
        patches[l].d_bias += eta * db.to(block.ff.fc3.weight.dtype)

    return patches
```

## 6. What was implemented in `nanoqwen/model.py`

The required model changes are now implemented in [`nanoqwen/model.py`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py):

1. [`FeedForward.forward`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py#L120) accepts an optional thought patch and can return an internal trace.
2. [`TransformerBlock.forward`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py#L241) accepts an optional layer patch and can return block traces.
3. [`Qwen3Model.forward`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py#L287) accepts `thought_patches` and `return_trace=True`.
4. [`generate_text_simple`](/Volumes/SB-XTM5/flair/software/continual-learning/nanoqwen/model.py#L592) can now generate with patched weights.
5. Local checkpoint loading is supported directly when `tokenizer.json` and `model.safetensors` already exist in a local directory.

The per-layer record only needs:

- `att_resid`
- `mlp_in`
- `mlp_hidden`
- `mlp_out`

At inference time, the patched MLP behaves like:

```python
gate = F.linear(x, self.fc1.weight + patch.d_fc1)
up = F.linear(x, self.fc2.weight + patch.d_fc2)
hidden = F.silu(gate) * up
out = F.linear(hidden, self.fc3.weight + patch.d_fc3)
if patch.d_bias is not None:
    out = out + patch.d_bias
```

This keeps the base Qwen checkpoint intact while making the paper's thought patch explicit.

## 7. Relation to the existing code in this repo

[`agent_tool_distill.py`](/Volumes/SB-XTM5/flair/software/continual-learning/agent_tool_distill.py#L1016) already implements the simpler rank-1 approximation:

```text
delta_a = a_teacher - a_vanilla
u = W delta_a
v = a_vanilla / ||a_vanilla||^2
DeltaW_i = u v^T
```

That is close to `learning-without-training.pdf`, but it is still narrower than the full Qwen thought-patching recipe above:

- it targets one linear by default
- it does not split the gated MLP into two first-layer solves
- it does not learn the output-side thought vector
- it compresses outer products with SVD instead of solving the paper's batch least-squares problem directly

So it is a good baseline, but not yet the paper's full algorithm for Qwen.

## 8. Recommended implementation order

1. Run [`qwen_thought_patch_cli.py`](/Volumes/SB-XTM5/flair/software/continual-learning/qwen_thought_patch_cli.py) on `multiply` first.
2. Inspect the per-step table and make sure patched eval accuracy rises while the contextual baseline stays strong.
3. Then run both `multiply,sum`.
4. Only after that, try more open-ended tasks like translation or new knowledge injection.

If the multiplication task fails, do not move on. That task is the shortest path to verifying that:

- token alignment is correct
- layer traces are correct
- the ridge solver is wired correctly
- the patch is being applied at the right point in the block
