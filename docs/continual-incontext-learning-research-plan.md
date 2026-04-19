# Continual Meta In-Context Learning: Research Plan

Date: 2026-04-19

## 0. One-paragraph summary

We train a small **update generator** `Uψ` on top of a frozen base LLM `fθ`
so that, given any context `C`, it produces a small persistent weight update
`Δθ = Uψ(C)`. The update is judged not by how well it reproduces the in-context
behavior on `C`, but by how much it improves the patched model `f_{θ+Δθ}`
on **held-out queries from the same hidden task with the context removed**.
Repeated across episodes drawn from a wide task family, this trains the model
to **compress lessons from context into weights**, rather than relying on
context to keep being there. We anchor the architecture on Dherin et al. 2025
(exact rank-1 implicit MLP update from a context) and Mazzawi et al. 2025
(token-independent thought patches by least squares), and we evaluate on a
cellular-automaton meta-learning suite plus the existing
[arithcl_benchmark](references/arithcl_benchmark) for compositional transfer.

## 1. Why this is worth doing now

Three recent results converge:

1. **Dherin et al. 2025** ([learning-without-training.pdf](learning-without-training.pdf)).
   For any contextual block (self-attention + first MLP), there is an *exact*
   token-dependent rank-1 update `Δ_xW(C) = (W·δA_x(C))·A(x)^T / ||A(x)||²`
   to the first MLP weight such that `T_{W+Δ_xW(C)}(x) = T_W(C, x)`. The full
   context's effect on the forward pass is literally a tiny weight patch.
2. **Mazzawi et al. 2025/2026** ([transmuting-prompts-into-weights.pdf](transmuting-prompts-into-weights.pdf)).
   These per-token patches can be *aggregated* into a single token-independent
   *thought vector* `δ(I) = mean_i δ_i` and *thought matrix*
   `Δ(I) = argmin_M Σ ||M·a_i − W·δ_i·a_i^T/||a_i||²·a_i||²`, which when
   applied to MLP weights reproduces full-context behavior on average for that
   instruction. This is empirical (Algorithm 1, Table 1: 100% on filtered
   arithmetic with patched-no-context Gemma-3 1B).
3. **Goldwaser et al. 2025**: the equivalence extends to modern blocks
   (Gemma-style, RMSNorm, gated MLP, MoE), so the mechanism is not a toy
   artifact.

What is *missing* is a learned, generalizable mapping from context to update.
Mazzawi's algorithm fits a fresh patch for each instruction by solving a
least-squares problem over a known calibration set. It does not amortize:
each new context requires its own labeled set, and there is no guarantee that
the patch transfers to inputs that look unlike the calibration distribution.

We propose to **train** `Uψ` to produce good `Δθ`s end-to-end, from
*unlabeled or partially-labeled* contexts, with the loss measured on
held-out queries. This converts the closed-form per-instruction recipe into
an inference-cheap, amortized continual learner.

This is also distinct from MAML and learned optimizers in two ways. (a) We do
not take inner gradient steps; the update is produced in one feed-forward
pass through `Uψ`. (b) We constrain the update to live in the low-rank MLP
patch space identified by Dherin and Mazzawi, which gives a strong, principled
inductive bias for what `Δθ` should look like.

## 2. Problem formulation

### 2.1 Episode

A meta-episode is a tuple `(τ, S, Q)`:

- `τ`: a hidden task drawn from a task family `T`.
- `S = (s_1, ..., s_n)`: a *support context* — a sequence of tokens that
  reveals enough about `τ` to do well on it. For CA, this is a list of
  `(state, next_state)` transitions under rule `τ`. For arithmetic, this is
  worked examples or an instruction plus a few examples.
- `Q = {(x_j, y_j)}`: a *held-out query set* under the same `τ`, with
  inputs `x_j` that *do not appear in `S`*.

The learner sees `S`, produces `Δθ = Uψ(S)`, and then must answer every `Q`
query *without `S` in context*. Only `Q` accuracy under
`f_{θ+Δθ}(·)` enters the loss.

### 2.2 What we are optimizing

```text
ψ* = argmin_ψ E_τ E_{(x,y)~Qτ}
       L( f_{θ + Uψ(Sτ)}(x), y )
       + auxiliary regularizers (Sec. 5)
```

`θ` is frozen everywhere — only `ψ` updates during meta-training. This is
critical: it preserves the substrate's general capability, and it makes the
`Δθ` head a clean, swappable module rather than an architectural rewrite.

### 2.3 The four anti-cheat constraints

To prevent `Uψ` from collapsing to "memorize the support tokens and predict
them on the query":

- **Disjoint support/query**: `x ∈ Q` is sampled fresh, never appearing in `S`.
- **Many queries per update**: `|Q| ≥ 32` queries share the same `Δθ`. A
  copying solution helps at most one.
- **OOD query split**: a fraction of queries are harder than support
  (longer state, deeper rollout, more terms). Forces rule abstraction.
- **Held-out task split**: a fraction of meta-test rules are never seen at
  meta-train (e.g., CA rules 0–199 train, 200–255 test).

## 3. Architecture

### 3.1 Three architectural variants, ordered by ambition

| Variant | What `Uψ` returns | When to try |
| --- | --- | --- |
| **V1: Closed-form patch** | Run the Mazzawi algorithm on `S` itself. No `Uψ` parameters. | Sanity-check: does the patch space contain a useful update for our episodes at all? Reuse `qwen_thought_patch_cli.py`. |
| **V2: Amortized patch generator** | A small transformer `Uψ` reads `S`, outputs per-layer `(δ_l, Δ_l)` directly. The patch shape matches Mazzawi's: a vector and a low-rank matrix per MLP block. | Main proposed method. |
| **V3: Two-timescale learner** | V2 plus a learned write-gate `Gψ` that decides which patches survive past one episode and accumulate into a maintained `Δθ_t` via decay-EMA. Trained with the multi-chunk meta-objective from §6.5. | Only after V2 works on three rule families. The full deployment-time state machine for V3 is specified in §6. |

We default to V2.

### 3.2 V2 in detail

```text
input:  S (token sequence, length up to N_ctx)
        per-layer activations from a "calibration" forward pass on S
        (cached once: a_l, h_l, mlp_in_l, mlp_out_l for each block l)

network Uψ:
   1. encode S with a small frozen pretrained encoder (e.g. last 2 layers
      of fθ as a feature extractor, or a separate ~10M-param encoder)
   2. for each block l in the patched layers:
        z_l = pool(per-token activations + S embeddings)
        (δ_l, Lψ_l, Rψ_l) = MLP_head_l(z_l)
        Δ_l = Lψ_l @ Rψ_l^T          # low-rank, rank r∈{1,2,4}
   3. patch the model:
        W_fc1_l ← W_fc1_l + Δ_l      (or fc1/fc2 separately for SwiGLU)
        and add δ_l to the MLP output (absorbed into fc3 row-space if no bias)

output: a LayerThoughtPatch list (same dataclass already in nanoqwen)
```

Patched layers default to the *middle third* of the stack. The tail is more
specialized; the head encodes generic features. (Empirically: Dherin's exact
construction is per-block; Mazzawi patches every block; we'll ablate.)

`Uψ` parameter count target: 5–20M for a 0.6–1.7B base. Small enough that
training is cheap; big enough that the head has capacity to encode 256+ CA
rules.

### 3.3 Bias-vs-low-rank-matrix decomposition

Following Dherin's two-part formula and Mazzawi's split:

- The **vector part `δ_l`** absorbs the *additive* shift in residual stream
  between with-context and without-context forward passes. For Qwen-style
  no-bias MLPs, fold it into the `fc3.weight` row mean (see
  [docs/thought-patch-agent-handoff.md](docs/thought-patch-agent-handoff.md)
  task 2 for the mechanism).
- The **matrix part `Δ_l`** is rank-r and parameterized as `Lψ Rψ^T`. We
  let r ∈ {1, 2, 4} and ablate.

This is the same shape Mazzawi proves is optimal under spherical activations.
It also matches the LoRA convention so `Uψ` can be swapped with off-the-shelf
LoRA tooling in the second half of the project.

## 4. The cellular-automaton meta-learning substrate

### 4.1 Why elementary CAs are the right warmup task

- Each episode has a *short*, *exact*, *enumerable* hidden rule (one of 256).
- Generalization is unambiguous: applying the rule to any new state is
  unambiguous and verifiable.
- Support set size and query OOD difficulty (length, rollout depth) are
  trivially controllable.
- Pretraining contamination is minimal — LLMs see rule numbers in text but
  do not internalize the rules.

### 4.2 Episode generator spec

File: `meta_icl/data/eca_episodes.py` (new).

```python
@dataclass(frozen=True)
class ECAEpisode:
    rule:        int                 # 0..255
    support:     list[tuple[str, str]]   # (state, next) pairs, length N_S
    query_id:    list[tuple[str, str]]   # same length distribution as support
    query_ood:   list[tuple[str, str]]   # 4x length, or 3-step rollout
    metadata:    dict[str, Any]
```

Defaults:

- `N_S = 16` support transitions, support state length L_s = 16
- `|Q_id| = 64` queries at L_s
- `|Q_ood| = 64` queries at L_s × 4 *or* 3-step rollout (the harder of the two
  randomly per episode)
- Rules 0–199 are meta-train; 200–255 are meta-test (fully held out).
- Within meta-train, 80/20 inner train/val split *of episodes*.

The support is rendered into a single prompt block. We use *two* templates per
training run, randomized per episode, to discourage prompt overfitting:

```text
template A (compact):
  "Rule transitions:
   01101 -> 11110
   10011 -> 00111
   ...
   Apply same rule to:
   <query>"

template B (sentence):
  "Below are state transitions under one elementary CA rule:
   given 01101 the next state is 11110.
   given 10011 the next state is 00111.
   ...
   What is the next state of <query>?"
```

For queries the *prefix* is just `Apply same rule to:` (template A) or
`What is the next state of <query>?` (template B). The support block does
*not* appear in the query prompt — that is the whole point.

### 4.3 Why this is a strict generalization of Mazzawi's setup

Mazzawi fits a patch per *instruction* (e.g., "Multiply numbers."). We let
the "instruction" be the support set itself, drawn from a continuous family
of rules. The amortized `Uψ` must learn to map *any* support-set shape to
the right patch — which is the exact thing the closed-form solver cannot do.

## 5. The training objective

We combine four loss terms. Coefficients are starting points; tune by ablation.

### 5.1 Held-out query loss (primary)

```text
L_query = E_{(x,y)~Q}  CE( f_{θ + Uψ(S)}(x), y )
```

This is the single non-negotiable term.

### 5.2 In-context teacher distillation

The teacher is the same base model conditioned on the support:
`p_teacher(·|x) = fθ(S, x)`. The student is the patched model with no
support: `p_student(·|x) = f_{θ + Uψ(S)}(x)`. Distill:

```text
L_distill = E_{x~Q}  KL( stop_grad(p_teacher(·|x)) || p_student(·|x) )
```

This anchors `Uψ` to "do whatever the in-context model does, without the
context." It is the cleanest operationalization of Dherin's
`T_W(C,x) = T_{W+ΔW}(x)` equivalence — the patched-no-context model should
behave like the unpatched-with-context model for the *same query*.

Coefficient: `λ_distill = 1.0` initially. Drop to 0.3 once `L_query` plateaus
to avoid teacher hallucinations dragging the student down on hard rules.

### 5.3 Improvement-over-raw margin

```text
L_raw    = CE( fθ(x), y )                       # raw model, no patch, no context
L_after  = CE( f_{θ + Uψ(S)}(x), y )            # patched, no context
L_margin = max(0, m + L_after - L_raw)
```

Forces the patch to actually *improve* over the unpatched model. Crucial in
early training when `Uψ` outputs near-zero patches by default and `L_query`
alone has no signal. `m = 0.05`. Drop after ~5k meta-steps.

### 5.4 Retention / anti-drift

Sample a batch of *general* prompts unrelated to `τ` (e.g., 32 short
WikiText snippets). Patched model logits on these should be close to raw
model logits:

```text
L_retain = E_{x~general}  KL( fθ(x) || f_{θ + Uψ(S)}(x) )
```

This protects against `Uψ` learning to scribble destructive patches that
help on `Q` but break the base model. Coefficient: `λ_retain = 0.5`.

### 5.5 Update bottleneck

Two regularizers on the produced `Δθ`:

```text
L_norm = ||δ_l||_2² + ||Δ_l||_F²       (sum over patched layers l)
L_rank = nuclear_norm(Δ_l)             (only if rank is not hard-capped)
```

We hard-cap rank by parameterizing `Δ_l = Lψ Rψ^T`, so `L_rank` is
typically zero. Keep `L_norm` with `λ_norm ∈ [1e-4, 1e-3]`.

### 5.6 Full loss

```text
L = L_query
  + λ_distill · L_distill
  + λ_margin  · L_margin
  + λ_retain  · L_retain
  + λ_norm    · L_norm
```

Default: `λ_distill = 1.0, λ_margin = 0.5, λ_retain = 0.5, λ_norm = 3e-4`.

### 5.7 Permutation-invariance auxiliary

Twice per batch, sample a permutation π of the support transitions and
require `||Uψ(S) − Uψ(π(S))||_F² ≤ ε`. Either as a hard architectural choice
(use a set transformer for the encoder) or as a soft loss with `λ_perm = 0.1`.
Prevents memorizing positional accidents in the support sequence.

### 5.8 Gradient-flow contract

This section nails down *what is differentiated* and *how the patch enters
the autograd graph*. It is the single piece most likely to silently break
the whole pipeline.

#### 5.8.1 What is the loss actually computed on

The patched model is judged by **teacher-forced cross-entropy of the gold
answer tokens on held-out queries**, never by greedy generation. `argmax` is
non-differentiable; generation accuracy is an *eval* metric only.

For each query `(x, y)`:

```text
prompt        = render(x)                # no support set in here
gold_ids      = tokenize(y)              # fixed-length, e.g. CA next-state
logits        = patched_forward(prompt + gold_ids[:-1])   # teacher-forced
L_query(x,y)  = mean CE(logits[answer_positions], gold_ids)
```

A batch of 64 queries shares one `Δθ`, so the loss is averaged over them —
the patch only wins if a single update helps many unseen queries.

For the distillation term `KL(p_teacher || p_student)`: compute
`p_teacher = fθ(S, x)` under `torch.no_grad()` and detach. Only the student
side carries gradient.

#### 5.8.2 The autograd-correct way to apply Δθ

The most common mistake is mutating the weight in place:

```python
# WRONG — breaks the graph, ψ gets no gradient
model.layers[l].mlp.fc1.weight.data += delta_l
```

`.data +=` writes into a leaf tensor with no grad history, so the patch
becomes invisible to autograd. Replacing the `nn.Parameter` with a fresh one
has the same problem.

The fix is to make the patch a **tensor that flows through the forward as an
argument**. Two equivalent options.

**Option A — `torch.func.functional_call` (cleanest, stateless):**

```python
def patched_logits(fθ, base_state, deltas, input_ids):
    # base_state: dict of name -> Parameter (frozen, no grad)
    # deltas:     dict of name -> tensor with grad linked to ψ
    overrides = {name: base_state[name] + deltas[name] for name in deltas}
    full_state = {**base_state, **overrides}
    return torch.func.functional_call(fθ, full_state, (input_ids,)).logits
```

The override-dict additions live in the autograd graph, so `loss.backward()`
walks `loss → logits → overrides[name] → deltas[name] → Lψ, Rψ, δ heads → ψ`.
fθ's parameters are leaves with `requires_grad=False`; only ψ accumulates
gradient.

**Option B — patch-aware forward (what `nanoqwen` already does):**

Add a `thought_patches` argument to the model's forward. Inside each block:

```python
def forward(self, x, patch=None):
    g = self.fc1.weight if patch is None else self.fc1.weight + patch.d_fc1
    u = self.fc2.weight if patch is None else self.fc2.weight + patch.d_fc2
    h = F.silu(F.linear(x, g)) * F.linear(x, u)
    out_w = self.fc3.weight if patch is None else self.fc3.weight + patch.d_fc3
    out = F.linear(h, out_w)
    if patch is not None and patch.d_bias is not None:
        out = out + patch.d_bias
    return out
```

The `+ patch.d_fc1` is a tensor add inside the graph. Gradient flows back
through it. This is the same path described in
[docs/thought-patching-qwen.md](docs/thought-patching-qwen.md) §6 — except
in that doc patches are *fit by least squares* (no autograd), and here we
*learn* them with autograd.

#### 5.8.3 End-to-end gradient chain for one episode

```python
# 1. Encode the support set into a context vector (fθ params not trained)
zS = Uψ_encoder(S_token_ids)             # zS: [d_z]

# 2. Per patched layer l, produce a low-rank factorization + bias vector
patches = []
for l in PATCHED_LAYERS:
    L_l = Uψ_L_head[l](zS).reshape(rank, d_out)   # rank-r factor
    R_l = Uψ_R_head[l](zS).reshape(rank, d_in)
    d_l = Uψ_bias_head[l](zS)                     # δ_l vector
    delta_W = L_l.transpose(0, 1) @ R_l           # rank ≤ r
    patches.append(LayerThoughtPatch(d_fc1=delta_W, d_bias=d_l, ...))

# 3. Run patched forward on every query, teacher-forced on the gold answer
batch_logits = patched_forward(fθ, patches, query_input_ids)   # [B, T, V]

# 4. Compute the differentiable loss
L_query = F.cross_entropy(
    batch_logits[:, answer_positions].reshape(-1, V),
    gold_ids.reshape(-1),
)
with torch.no_grad():
    p_teacher = F.softmax(
        fθ(S_then_query_ids).logits[:, answer_positions], dim=-1
    )
p_student = F.log_softmax(batch_logits[:, answer_positions], dim=-1)
L_distill = F.kl_div(p_student, p_teacher, reduction="batchmean")

with torch.no_grad():
    L_raw = F.cross_entropy(
        fθ(query_input_ids).logits[:, answer_positions].reshape(-1, V),
        gold_ids.reshape(-1),
    )
L_margin = F.relu(0.05 + (L_query - L_raw))     # only L_query has grad

L_retain = F.kl_div(...)                        # patched vs raw on unrelated text
L_norm   = sum(p.d_fc1.pow(2).sum() + p.d_bias.pow(2).sum() for p in patches)

loss = (
    L_query
    + 1.0 * L_distill
    + 0.5 * L_margin
    + 0.5 * L_retain
    + 3e-4 * L_norm
)
loss.backward()
opt_ψ.step()
```

The factorization `delta_W = L_l.T @ R_l` lives outside `Uψ`. `Uψ` only
emits the thin factors. Gradient flows from `delta_W` back through both
`L_l` and `R_l` linearly.

#### 5.8.4 Required smoke tests (`tests/test_gradient_flow.py`)

These are not optional — they catch every common breakage in this pipeline.

1. **ψ gets gradient, fθ does not.** After `loss.backward()`, every
   parameter of `Uψ` has a non-None `.grad` and at least one entry is
   non-zero. Every parameter of `fθ` has `.grad is None` (or is exactly
   zero), confirming the substrate is frozen.
2. **Patch is in the graph.** `delta_W.requires_grad is True` and
   `delta_W.grad_fn is not None`.
3. **Zero-patch identity.** With `patches = [zeros for l in PATCHED_LAYERS]`,
   the patched forward equals the raw forward to within float noise. Catches
   accidental hidden-state leakage from `S` into the patched pass.
4. **Permutation invariance** (if claimed). With `S` randomly permuted,
   `Uψ(S)` may change, but `loss` should not change beyond a tight
   tolerance. Catches positional cheating.
5. **Support/query disjointness check.** If a query input also appears in
   the support, a copy-only `Uψ` could win on that query. The episode
   sampler must reject overlap. Test by injecting one overlap and asserting
   the episode is rejected.

#### 5.8.5 Three real-world traps

- **Greedy decoding for L_query.** Tempting because exact match is the
  headline metric. Don't. `argmax` zeroes out the gradient. Train on
  teacher-forced CE; report greedy accuracy only at eval.
- **Forgetting `no_grad` on the teacher.** If `p_teacher` is not detached,
  it accumulates `Uψ`-gradient via the shared embeddings. With fθ frozen
  the effect is zero, but if you later unfreeze any fθ piece you will
  silently train it on the wrong objective.
- **Activation memory.** Backprop through 24 transformer layers × 64
  queries × 32 answer tokens with the patch tensor in the graph blows up
  memory. Apply gradient checkpointing on fθ blocks, or cap the gradient
  batch at 8–16 queries per step (the eval `|Q|` stays at 64+, so
  copy-cheating is still blocked — only the gradient is computed on a
  sub-batch).

## 6. Inference-time continual learning protocol

Sections 1–5 describe meta-training. This section describes what runs at
**deployment**, when there is no labeled query set and tokens arrive over
time. There are three distinct modes the trained system can run in. Pick
the mode based on the use case; do not conflate them.

### 6.1 Three deployment modes

| Mode | What runs | Use case |
| --- | --- | --- |
| **M1: one-shot adaptation** | Given a complete `S`, compute `Δθ = Uψ(S)` once, then answer queries from `f_{θ+Δθ}`. No state, no updates after. | The Mazzawi-style "instruction-as-weights" replacement of ICL. Validated by Phase 2 directly. |
| **M2: streaming continual inference** | Maintain `Δθ_t` as evolving state. Update at chunk boundaries via gated EMA. **No gradients.** | A long-running agent or assistant accumulating knowledge from interactions. Validated by Phase 4. |
| **M3: permanent consolidation** | Periodically promote `Δθ_t` into base weights `θ ← θ + α·Δθ_t` after extensive evidence and replay. | Out of scope until M2 is stable. |

Critically, **`Uψ` and `Gψ` are frozen at inference in all three modes**.
The only "learning" that happens at deployment is the maintained `Δθ_t`
state in M2 and the rare base-weight write in M3. There is no SGD or
backprop at runtime.

### 6.2 The M2 state machine

State the deployed system carries:

```text
θ           — base weights, never modified
Δθ_t        — accumulated persistent patch; the "memory"
buffer_t    — bounded FIFO of recent salient chunks (~64 chunks,
              each ~512 tokens; oldest dropped)
```

At each chunk boundary (every `K` generated tokens, or every turn):

```python
@torch.no_grad()
def step(self, new_chunk):
    self.buffer.append(new_chunk)
    Δθ_candidate = self.Uψ(list(self.buffer))         # one Uψ forward pass
    g = self.gate(Δθ_candidate, new_chunk)            # ∈ {0, 1} or [0, 1]
    for l in PATCHED_LAYERS:
        self.Δθ[l] = self.decay * self.Δθ[l] + g * Δθ_candidate[l]
        self.Δθ[l] = project_to_norm_ball(self.Δθ[l], radius=R)
```

Generation always uses `f_{θ + Δθ_t}` plus whatever fits in the regular
context window. Recent material stays in attention; older material lives in
`Δθ_t`. The split between the two is a hyperparameter (chunk size + buffer
length).

### 6.3 The gate `Gψ` without labels

At deployment there is no teacher. `Gψ` uses two self-supervised signals
that can be computed from the chunk itself plus a small fixed retention set
loaded at init:

```python
@torch.no_grad()
def gate(self, Δθ_new, chunk):
    # 1. Helpfulness: does the candidate better reproduce what just happened?
    loss_old = tf_ce(self.fθ, self.Δθ,                  chunk)
    loss_new = tf_ce(self.fθ, _add(self.Δθ, Δθ_new),    chunk)
    helps = (loss_old - loss_new) > MIN_DROP

    # 2. Safety: does it preserve unrelated knowledge?
    drift = (
        tf_ce(self.fθ, _add(self.Δθ, Δθ_new), RETENTION_PROBES)
        - tf_ce(self.fθ, self.Δθ,             RETENTION_PROBES)
    )
    safe = drift < MAX_DRIFT

    return 1.0 if (helps and safe) else 0.0
```

`RETENTION_PROBES` is the deployment-time analogue of `L_retain` from §5.4
— a fixed set of ~32 short unrelated text snippets loaded once at init.

The hard `{0, 1}` gate above is the simplest version. A learned soft gate
`g = σ(MLP_Gψ(features))` taking the same features as input is a strict
generalization and is what V3 should train (§3.1, §9 Phase 4).

### 6.4 Patch combination strategy

Three options, in increasing complexity:

- **Replace.** `Δθ_{t+1} = Uψ(buffer_{t+1})`. Simple, but information older
  than the buffer is gone.
- **Sum + decay (EMA).** As in 6.2. Information decays gradually; new
  information can interfere additively.
- **Skill library.** Maintain `N` named patches `{Δθ^{(1)}, ..., Δθ^{(N)}}`
  with a router that selects or gate-sums at decode time. Composable across
  unrelated tasks but expensive at inference.

Default for Phase 4 is **sum + decay** for two reasons: it is the simplest
non-trivial design, and it falls out naturally if `Δθ_t` is treated as a
fast-weight memory à la Titans. Skill library is a Phase 4+ extension only
if EMA suffers measurable interference between unrelated tasks.

### 6.5 Why the §5 meta-training does not by itself produce a good M2

If the §5 episodes are all single-shot `(S, Q)`, `Uψ` learns to map *a
complete support set* to a patch. Deployed in M2, the same `Uψ` will be
asked to produce patches that **compose with an existing `Δθ_{t-1}`**. It
was never trained for this. The deployed EMA will drift or destructively
overwrite.

The fix is to add **multi-chunk meta-training episodes** to the §5 mix
(say, 25% of episodes once V2 single-shot is stable):

```text
chunks         = [c_1, c_2, ..., c_T]    # split support into T contiguous pieces
Q_progressive  = [Q_1, Q_2, ..., Q_T]    # held-out at each prefix
state          = (Δθ_0 = 0, buffer = [])
for t in 1..T:
    Δθ_candidate = Uψ(buffer + [c_t])
    g            = Gψ(features_t)        # learnable scalar gate
    Δθ_t         = decay * Δθ_{t-1} + g * Δθ_candidate
    L           += λ_t * L_query(f_{θ + Δθ_t}, Q_t)
    buffer       = update(buffer, c_t)
loss = L     # backprop through all T steps
```

This trains both `Uψ` and `Gψ` end-to-end to produce *composable* patches
and to gate them correctly. Gradient flows through the EMA the same way it
flows through any RNN — backprop through time. With `T = 4` and gradient
checkpointing on each `f_{θ + Δθ_t}` forward, memory is manageable.

`Q_t` should test what *should* be known after seeing `c_1..c_t`, not the
final task — otherwise early steps are unsupervised noise. For ECA: `Q_t`
queries a sub-rule that is fully determined by the union of neighborhoods
seen in `c_1..c_t`.

### 6.6 Failure modes specific to deployment

- **`Δθ` explodes.** Any unconstrained EMA drifts. Project to a fixed
  Frobenius ball after each step (`R` chosen so single-shot M1 evaluation
  just barely fits inside it). Monitor `||Δθ_t||_F` over a long session;
  it should plateau, not grow unboundedly.
- **Catastrophic forgetting in patch space.** Later patches overwrite
  earlier ones in the same low-rank subspace. Lightweight fix: per-layer
  norm caps + decay. Heavyweight fix: skill library (§6.4).
- **Self-supervised gate is brittle.** It can rubber-stamp every update
  or reject every update depending on threshold. `MIN_DROP` and
  `MAX_DRIFT` are hyperparameters tuned on a held-out *streaming* eval
  (§6.7), not on single-shot accuracy.
- **Buffer-vs-context confusion.** The patched model sees both `Δθ_t` and
  a fresh context window during generation. The gate's helpfulness probe
  must evaluate the chunk *without recent context*, otherwise context
  contamination makes every patch look helpful.

### 6.7 Streaming evaluation harness (Phase 4 deliverable)

Single-shot CA accuracy does not test M2. We need a separate evaluation
harness:

```text
streaming_episode:
  rule_sequence = [r_1, r_2, ..., r_E]   # E ≤ 8 different rules in order
  for e in 1..E:
    deliver chunks of rule r_e to the deployed system
    after every D chunks, evaluate on:
      - Q_current(r_e)              # current rule
      - Q_recent({r_{e-1}, r_e})    # immediate retention
      - Q_old(r_1)                  # long-term retention
      - Q_unrelated(general text)   # no-drift check
```

Metrics on this harness:

- **Adaptation lag**: chunks needed before `Q_current` accuracy crosses a
  fixed threshold after a rule switch.
- **Retention curve**: `Q_old(r_1)` accuracy over time as more rules
  arrive. The slope is the forgetting rate.
- **Interference**: drop in `Q_unrelated` accuracy. Should be zero.
- **Patch-norm trajectory**: `||Δθ_t||_F` over time. Should be bounded.

This harness reuses the ECA episode generator (§4.2) plus a thin streaming
wrapper. Phase 4 succeeds only if all four metrics behave reasonably; Phase
4 fails if the M2 deployment trades current accuracy for retention or vice
versa in an uncontrollable way.

## 7. Baselines we must compare against

Each baseline is a different strategy for handling support. They make the
"Uψ trained on outer-loop transfer" story testable.

| Baseline | What it does | Expected behavior |
| --- | --- | --- |
| **B0: Raw** | Ignore `S`. `f_θ(Q)`. | Floor; ICL/abstraction gap. |
| **B1: ICL** | Run `f_θ(S, Q)` with full context. | Ceiling we want to match without the context. |
| **B2: Mazzawi closed-form** | Run thought-patch fitting on `S` per episode (no `Uψ`). | The training-free version of the proposed method. The whole point of `Uψ` is to beat this on speed and/or held-out generalization. |
| **B3: Per-episode SGD inner-loop (MAML-ish)** | At test, take `k` SGD steps on the support, then evaluate on query. | Compute-expensive baseline. If `Uψ` matches it with one forward pass, that is the core wins claim. |
| **B4: LoRA-from-scratch per episode** | Initialize and fit a small LoRA on `S` from scratch, k steps. | Disentangles "learned-Uψ" from "any low-rank update." |
| **B5: Vanilla finetuning of fθ on Q** | Cheating oracle: train on `Q` directly. | Upper bound on how good the patched-no-context model could be. |
| **B6: Task vectors (Hendel et al. 2023)** | Sum activations over `S`, add as steering vector. | The vector-only sibling of our matrix patch. |

Headline result: V2 (`Uψ`) should match or beat B2 *and* B3 on `Q_ood` for
held-out rules (Sec. 4 split), at < 5% the inference cost of B3.

## 8. Compositional and continual evaluation via ArithCL

The CA suite tests rule abstraction in isolation. To test the *continual*
half — does compressing context into weights *help future learning* — we
hook the trained `Uψ` into the [arithcl_benchmark](references/arithcl_benchmark)
runner.

The wrapper is a `BaseLearner` whose `update(batch)` *does not change `θ`*.
Instead it caches each batch as a growing context `S_t`, runs
`Δθ_t = Uψ(S_t)`, and `predict()` uses `f_{θ+Δθ_t}(·)`. Evaluation pieces:

- `three_stage_arith_v1`: addition → multiplication → linear combination.
  Hypothesis: the patch produced after seeing addition examples should
  *improve* zero-shot accuracy on linear combination (positive forward
  transfer, the FWT metric the benchmark already computes).
- `extended_arith_v1`: tests deeper composition.
- `warmup_transfer_v1`: tests whether ECA-pretrained `Uψ` (Sec. 4) gives
  better arithmetic continual learning than CA-cold `Uψ`. This is the
  cleanest test of the "abstract pretraining transfers" hypothesis discussed
  in `docs/in-context-learning-literature-notes.md`.

We extend `arithcl_benchmark` with one new file
`arithcl_benchmark/baselines_metaicl.py` exposing the wrapper. No changes to
the benchmark core.

Metric we will report on top of the benchmark's defaults: **zero-shot
composition accuracy** on linear combination *before* its stage starts, after
the addition+multiplication stages have populated `S`. This is the headline
"better at later learning because of earlier learning" claim.

## 9. Phased roadmap

Each phase has a hard pass/fail gate. Do not start phase N+1 until N's gate
is green.

### Phase 0: Reproduce Mazzawi (baseline B2), 1 week

- Confirm [`docs/thought-patching-qwen.md`](docs/thought-patching-qwen.md)
  results: filtered multiply ≥ 70% patched on Qwen3-0.6B, ideally → 100%
  with the fc3-absorption fix from
  [`docs/thought-patch-agent-handoff.md`](docs/thought-patch-agent-handoff.md).
- Add summation, French→English translation as in Mazzawi Table 3.
- **Gate**: filtered patched-no-context multiply ≥ 90% on Qwen3-1.7B.
  Without this we have no working substrate.

### Phase 1: V1 closed-form on ECA episodes, 1 week

- Implement `meta_icl/data/eca_episodes.py` (Sec. 4.2).
- Implement `meta_icl/closed_form_eca.py`: for each held-out rule, fit a
  thought patch from the support set the same way Mazzawi fits one from
  arithmetic prompts.
- Evaluate V1 on Q_id and Q_ood for held-out rules.
- **Gate**: V1 beats B0 (raw) by ≥ 20 pp on held-out Q_id, and > 5 pp on
  Q_ood. If the patch space cannot encode CA rules at all, V2 will not work
  either, so investigate substrate model choice / which layers to patch
  before continuing.

### Phase 2: V2 amortized `Uψ`, 3–4 weeks

- Implement `meta_icl/uψ_model.py` (Sec. 3.2).
- Implement `meta_icl/train.py` with the loss in Sec. 5 and an episode
  sampler that does the four anti-cheat steps (Sec. 2.3).
- Train on rules 0–199, evaluate on 200–255.
- **Gate**: V2 ≥ V1 on held-out Q_ood by ≥ 5 pp, *and* the inference cost
  per episode is ≤ 2 forward passes through `fθ` (`Uψ` itself is cheap).
- Sub-gate (sanity): V2 with the support-set shuffled per episode achieves
  ≤ 50% of unshuffled accuracy. This proves it is using support content,
  not surface artifacts.

### Phase 3: Plug into ArithCL, 1–2 weeks

- Implement `arithcl_benchmark/baselines_metaicl.py`.
- Run `three_stage_arith_v1` and `warmup_transfer_v1` with V2 as the
  learner. Compare to `composable_oracle` (Sec. 8).
- **Gate**: V2's zero-shot composition accuracy on `linear_combination`
  (before its stage) is non-trivial (> 10 pp above raw model). FWT
  positive on at least one split.

### Phase 4: V3 two-timescale + write gate, optional, 3+ weeks

Only if Phases 1–3 succeed.

- Add a learned scalar gate `g_t = σ(Gψ(loss_gap_t, support_consistency_t,
  patch_norm_t))`. Apply `Δθ_consolidated += g_t · EMA(Uψ(S_t))`.
- Test long episodes (100+ support items, 500+ tokens of context).
- **Gate**: long-episode patched performance is monotone non-decreasing in
  episode length on at least the held-out CA rules. If it degrades, the
  consolidation step is destructive and needs more retention pressure or
  smaller `g`.

## 10. Engineering plan

### 10.1 Repo layout

New top-level package `meta_icl/`:

```
meta_icl/
  data/
    eca_episodes.py        # Sec. 4.2 generator + dataset
    arith_episodes.py      # ArithCL episodes for direct use, Phase 3
  models/
    base_loader.py         # Wraps nanoqwen + thought-patch hooks
    encoder.py             # Support-set encoder; defaults to last 2 frozen
                           # layers of fθ + small set transformer
    uψ_head.py             # Per-layer (δ, Lψ, Rψ) heads; rank ∈ {1,2,4}
  losses.py                # L_query, L_distill, L_margin, L_retain, L_norm
  episode.py               # ECAEpisode dataclass + collation
  train.py                 # Meta-training loop
  evaluate.py              # Held-out eval harness
  closed_form.py           # V1 baseline; reuses qwen_thought_patch_cli logic
  cli.py                   # Single entrypoint with subcommands

tests/
  test_eca_episodes.py
  test_uψ_shapes.py
  test_losses.py
  test_gradient_flow.py    # the five checks in §5.8.4 — non-optional
  test_train_smoke.py      # 50-step run on 4 rules, smoke level only
```

We do *not* fold this into the existing SPCA package (`network.py`,
`neuron.py`, etc., per [CLAUDE.md memory](memory/MEMORY.md)) — that is a
gradient-free text-token system and shares no machinery. Keep the two
threads separate.

We *do* depend on `nanoqwen/model.py` (already supports
`ThoughtPatch`-style patched forward) and `references/arithcl_benchmark/`.

### 10.2 Compute budget

V2 meta-training:

- Base model: Qwen3-0.6B for fast iteration; Qwen3-1.7B for final results.
- Episode size: 16 support + 64 id-query + 64 ood-query.
- Per episode: 3 forward passes through `fθ` (teacher with context, student
  patched without context, raw without context) + 1 small `Uψ` forward.
- Batch size: 4 episodes per gradient step (so 4 × 144 = 576 sequences).
- 50k meta-steps as a starting target. On a single A100 with `bf16` plus
  KV-cache reuse for the context-shared teacher pass, estimated ~3 days.

Phase 3 / 4 add no significant additional compute.

### 10.3 Observability

For each meta-step log:

- per-loss-component value
- patch norms `||δ_l||, ||Δ_l||_F` per layer (mean and max)
- *patch alignment*: cosine between Mazzawi-closed-form patch on the same
  support and `Uψ`'s output. Drift here is informative: Uψ may discover
  better patches than the closed form, or it may diverge into
  un-interpretable updates that still help. Either is fine; we just need
  to know which.
- inference-time tokens per second for V2 vs B2 vs B3.

For each evaluation epoch save:

- query accuracy by rule, by split (id/ood), to a per-rule heatmap.
- a sample of 8 wrong predictions verbatim.

## 11. Ablations (the experiments that earn the paper)

Each isolates one design choice. Run in this order.

1. **Patch space**: rank ∈ {0, 1, 2, 4, 8}. r=0 means vector-only (steering).
   Hypothesis: r=2 is enough; r=8 overfits on held-out rules.
2. **Patched layers**: head-third / middle-third / tail-third / all.
   Hypothesis: middle-third dominates (matches Mazzawi observation that
   middle MLPs carry most of the steering effect).
3. **Loss components**: drop each of distill / margin / retain / norm in
   turn. Most informative ablation; isolates which signal is doing the work.
4. **Support-set size**: 4, 16, 64 transitions. Hypothesis: V2 is
   monotone-improving in `|S|` up to ~32, then plateaus.
5. **Anti-cheat**: turn off support/query disjointness. If accuracy stays
   the same on held-out rules, our anti-cheat was unnecessary; if it
   collapses, we caught real cheating.
6. **Encoder choice**: frozen-fθ-tail vs separate set-transformer. Whether
   the support encoder needs to see fθ's representations matters a lot for
   future scaling.
7. **Curriculum**: train on all 200 rules uniformly vs easy-first
   (parity, shifts, then chaotic 30/110). Speed of convergence and final
   held-out accuracy.

## 12. What we expect to see (and what would falsify the idea)

Positive outcome:

- V2 produces a Δθ in one forward pass that gets within 5 pp of B1 (full
  ICL) and beats B2 (closed-form) on ood queries for held-out rules.
- ArithCL: positive FWT on the composition stages compared to a raw
  base-model continual learner.
- The middle-layer rank-2 patch is small in norm (`||Δ_l||_F < 1.0`) and
  visibly task-coherent (e.g., for additive CA rules, similar Δs cluster).

Falsification — any of the following kills or reshapes the idea:

- V1 cannot beat raw on Q_ood for any layer choice and any rank ≤ 8.
  Means: the implicit-update space simply does not contain a useful CA
  rule encoder. Reshape: try a different substrate (small base model
  pretrained on synthetic structure), or move to vector-only steering.
- V2 trains, but its outputs are indistinguishable from B2's closed-form
  patch and add no value. Still a paper, but a much smaller one — we'd be
  showing that the closed-form is essentially optimal, and the contribution
  is amortization speed only.
- V2 helps on meta-train rules but does not generalize to held-out rules.
  Means: `Uψ` overfits the rule distribution, not the underlying pattern.
  Probably needs a larger and more diverse rule set (move beyond the 256
  ECAs to higher-order CAs or random Boolean circuits).
- Retention loss rises throughout training and the patched model becomes
  generally worse on WikiText. Means the patch space is fundamentally
  destructive at this scale; we need stronger constraints (smaller rank,
  fewer layers, gating).

## 13. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The exact-equivalence theorem is per-token and our amortized patch is per-task. Cross-token aggregation may lose too much. | Mazzawi already shows the aggregated form works on real instructions. We use exactly her form as our hypothesis class, with a learned producer in front. |
| Training-time forward passes are 3× single-context inference. | KV-cache the support pass once per episode (it is shared between teacher and the calibration trace `Uψ` consumes). Use bf16 throughout. |
| Held-out CA rules may share so much structure with training rules that we measure interpolation, not extrapolation. | Use the full 0–199 / 200–255 split as a cold test, *and* a structural-class split (e.g., train on rules with ≤2 active neighborhood bits, test on ≥3). Report both. |
| `Uψ` with a frozen-fθ encoder makes the system base-model-specific. | Phase 4 ablation 6 measures this. If it's bad, fall back to a separate encoder. |
| Engineering cost of plugging into nanoqwen vs HuggingFace transformers. | nanoqwen already exposes the patch hooks (per `docs/thought-patching-qwen.md` Sec. 6). Stick with it through Phase 2; only port to HF if a bigger base model is needed for Phase 3+. |

## 14. Connection to the other docs in this repo

- `docs/continual-incontext-learning-random-drafts.md`: this plan is the
  concrete, scoped version of that brain dump. The dump has additional
  ideas (attention-bias delta `Bψ`, three-timescale "living organism")
  which we treat as Phase 4+ optional extensions, not Phase 1 commitments.
- `docs/in-context-learning-literature-notes.md`: provides the citation
  context. The synthesis at the end of that doc — "the literature is still
  weak on how a system should judge whether an internal update is merely
  fitting the current prompt or actually discovering a higher-level
  principle worth stabilizing" — is the gap this plan targets directly.
- `docs/thought-patching-qwen.md` and `docs/thought-patch-agent-handoff.md`:
  these are the working substrate. Phase 0 success depends on the fc3
  absorption fix described in the handoff.
- `references/arithcl_benchmark`: provides the continual-learning
  evaluation in Phase 3 with no modification to the benchmark core.

## 15. First two weeks: concrete next steps

In order:

1. Read [docs/thought-patch-agent-handoff.md](docs/thought-patch-agent-handoff.md)
   and finish the Phase 0 fc3 absorption work if it is not already done.
   Verify Mazzawi's Table 1 numbers reproduce on Qwen3-1.7B.
2. Create `meta_icl/data/eca_episodes.py` with the dataclass and generator
   from Sec. 4.2. Add `tests/test_eca_episodes.py` covering: rule
   correctness, OOD split, train/test rule disjointness, two prompt
   templates.
3. Implement `meta_icl/closed_form.py` (V1) by lifting the patch-fitting
   logic from `qwen_thought_patch_cli.py` and feeding it ECA support sets
   instead of arithmetic instructions.
4. Run V1 on 30 held-out ECA rules. Decide on Phase 1 gate.
5. Before any V2 training run, implement `tests/test_gradient_flow.py`
   covering the five checks in §5.8.4. Treat a red test as a hard block
   on starting Phase 2.
6. Only then implement V2.

The two things most likely to silently invalidate the whole project: the
Sec. 2.3 anti-cheat episode discipline (support/query overlap or shared
templates make both V1 and V2 look great while teaching us nothing), and
the Sec. 5.8 gradient-flow contract (a misplaced `.data +=` or missing
`functional_call` means ψ never gets a real gradient and training looks
like it is working while learning nothing).
