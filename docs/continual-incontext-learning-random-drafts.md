You are pointing at the right abstraction: **the model should not merely use context; it should learn how to convert context into an update rule**.

The paper you linked, _Learning without training: The implicit dynamics of in-context learning_, is directly relevant, but it suggests one important correction to your framing:

> The most natural delta is probably **not the raw attention-score delta**.
> The paper’s core object is the **context-induced activation delta**, which can be converted into a **low-rank MLP weight update**.

Attention deltas may still be useful, but I would treat them as an auxiliary representation, not the main thing to optimize.

---

## 1. What the paper actually says

The paper defines a “contextual block” as:

```text
contextual layer A  →  MLP M_W
```

For a transformer block, `A` can be the self-attention layer, and `M_W` is the MLP after attention. The key quantity is:

```text
δA_x(C) = A(C, x) - A(x)
```

That is: the difference between the attention-layer output for query `x` **with context** and **without context**. The paper shows that this contextual effect can be rewritten as a **rank-1 update** to the first MLP weight matrix `W`. In the simplified setting:

```text
Δ_x W(C) = (W δA_x(C)) A(x)^T / ||A(x)||²
```

and then:

```text
T_W(C, x) = T_{W + Δ_x W(C)}(x)
```

So the model with context is equivalent, for that query, to the model without context but with a context-derived weight update. This is the crucial bridge from **ICL** to **persistent learning**. ([arXiv][1])

The paper then interprets the sequence of context tokens as an implicit learning trajectory. As more tokens are consumed, the induced weights move through a sequence:

```text
W0 → W1 → W2 → ... → Wn
```

and the authors show this can be viewed as an online gradient-descent-like process, where each token contributes a marginal update. ([arXiv][1])

Experimentally, they test this on in-context linear regression: a transformer sees examples `(x_i, h(x_i))` and predicts `h(x_query)`. They show that the contextual prediction and the prediction made by applying the derived `ΔW` without context agree up to numerical precision, and that the implicit update direction remains highly aligned with explicit finetuning updates. ([arXiv][1])

The most important limitation for your goal is this: the exact `Δ_x W(C)` is **query-dependent**. The paper explicitly says that the context cannot generally be represented exactly by one fixed static update for all possible future inputs, but suggests that aggregating token-dependent updates, for example by averaging, is a promising path toward prompt compression and context reuse. ([arXiv][1])

That is basically your research problem.

---

# 2. The right training objective

Your idea is:

```text
with context:    pθ(y | C, x)
without context: pθ(y | x)

difference = what is missing from the weights
```

That is right. But instead of directly regularizing:

```text
attention_score_augmented - attention_score_raw
```

I would train a **context-to-update module**:

```text
Uψ(C) → Δθ
```

where `Δθ` is a small persistent update, usually a low-rank adapter, MLP update, attention-bias update, or fast-weight memory.

Then evaluate:

```text
p_{θ + Δθ}(y | x)
```

on **held-out future inputs from the same hidden task**, with the context removed.

So the learning target is not:

```text
Can the model answer the current token with context?
```

It is:

```text
Can the model compress the lesson from context into Δθ,
so that it performs better later without the context?
```

That is the central move.

---

# 3. A concrete meta-learning episode

Use cellular automata as the task family.

Each episode samples a hidden rule `r`.

For example:

```text
rule r = unknown elementary cellular automaton
support set S = examples of transitions under r
query set Q = new states under the same r
```

Support:

```text
State:  0 1 1 0 1 0 0
Next:   1 1 1 1 0

State:  1 0 0 1 1 0 1
Next:   0 0 1 1 1

State:  0 0 1 1 0 1 0
Next:   0 1 1 1 1
```

Query:

```text
State:  1 1 0 0 1 0 0
Next:   ?
```

The learner receives `S`, produces an update `Δθ`, and then must answer the query **without seeing S again**:

```text
Δθ = Uψ(S)

prediction = f_{θ + Δθ}(query_state)
```

Loss:

```text
L_query = CE(f_{θ + Δθ}(x_q), y_q)
```

The update only gets rewarded if it improves performance on unseen query states. That prevents the trivial “copy the next token from context” solution.

---

# 4. The core loss

A good objective should combine four terms.

## 4.1 Query generalization loss

This is the main one:

```text
L_query = E_{τ} E_{(x,y)∈Qτ}
          CE(f_{θ + Uψ(Sτ)}(x), y)
```

where:

```text
τ = hidden task
Sτ = support context
Qτ = held-out query set
```

This trains the model to produce an update that helps on **new examples**, not merely on the support examples.

---

## 4.2 Improvement-over-raw loss

You want the update to beat the raw model:

```text
L_margin = max(0, m + L_after - L_raw)
```

where:

```text
L_raw   = CE(fθ(x), y)
L_after = CE(f_{θ + Δθ}(x), y)
```

So the update is rewarded only if it improves the future prediction.

This directly operationalizes your idea:

> the gap between in-context and no-context prediction tells us what is missing from the weights.

---

## 4.3 In-context teacher distillation

Use the full-context model as a teacher:

```text
p_teacher = fθ(Sτ, x_q)
p_student = f_{θ + Δθ}(x_q)
```

Then add:

```text
L_distill = KL(stopgrad(p_teacher) || p_student)
```

This says:

> whatever the model can do using context, learn to do it without context by writing the lesson into weights.

This is the cleanest way to “distill ICL into parameters.”

It is very close in spirit to the paper’s exact equality:

```text
T_W(C, x) = T_{W + Δ_xW(C)}(x)
```

but now we train a reusable, approximate, task-level `Δθ` instead of relying on a query-specific theoretical update. ([arXiv][1])

---

## 4.4 Retention / anti-drift loss

You also need to protect existing ability:

```text
L_retain = KL(fθ(x_general) || f_{θ + Δθ}(x_general))
```

on unrelated general data.

This prevents the update from becoming a destructive hack.

---

## Full objective

```text
L =
  L_query
+ λ_distill L_distill
+ λ_margin  L_margin
+ λ_retain  L_retain
+ λ_norm    ||Δθ||²
+ λ_rank    rank_penalty(Δθ)
```

That is the actual training recipe.

---

# 5. How to use the paper’s `ΔW` formula directly

The paper gives an exact **per-query** update:

```text
Δ_x W(C) = (W δA_x(C)) A(x)^T / ||A(x)||²
```

But you want one update that persists across many future inputs.

So do this:

For a support context `C`, sample a set of unlabeled probe queries:

```text
P = {x_1, x_2, ..., x_m}
```

For each probe, compute the desired MLP preactivation shift:

```text
t_i = W [A(C, x_i) - A(x_i)]
```

You want one static update `ΔW_C` such that:

```text
ΔW_C A(x_i) ≈ t_i
```

for many probes `x_i`.

So solve:

```text
ΔW_C* = argmin_Δ Σ_i ||Δ A(x_i) - W[A(C,x_i)-A(x_i)]||² + ρ||Δ||²
```

In matrix form:

```text
H = [A(x_1), ..., A(x_m)]
T = [t_1, ..., t_m]

ΔW_C* = T H^T (H H^T + ρI)^-1
```

Then project it to rank `r` with SVD:

```text
ΔW_C,r* = rank-r approximation of ΔW_C*
```

This gives you a **context-to-weight distillation target**.

Then train:

```text
Uψ(C) ≈ ΔW_C,r*
```

plus the query loss above.

This is the most principled version of your idea.

The paper gives the exact per-query implicit update; your method turns many per-query implicit updates into one approximate persistent update.

---

# 6. Where attention-score deltas fit

You proposed:

```text
input → raw attention scores
context + input → augmented attention scores

attention_delta = augmented - raw
```

This is reasonable, but attention scores are a fragile target.

Why? Because there are many attention patterns that can yield the same output, and the paper itself notes that the implicit update is not unique. ([arXiv][1])

So I would not train the model mainly to match attention scores. I would train it to improve held-out prediction after applying an update.

But you can add attention deltas as an auxiliary mechanism.

At layer `l`, head `h`:

```text
raw attention logits:
A_raw = QK^T / sqrt(d)

augmented attention logits:
A_aug = A_raw + Bψ(C, x)
```

where `Bψ(C, x)` is a learned attention-bias delta.

Then:

```text
Attn_aug = softmax(A_raw + Bψ(C, x))
```

Loss:

```text
L_attn = || Attn_aug_without_context(x)
          - stopgrad(Attn_with_context(C,x)) ||²
```

But the primary loss should still be:

```text
Does this delta improve future held-out queries?
```

Not:

```text
Does this delta imitate a particular attention map?
```

So the hierarchy should be:

```text
main target: future query loss
secondary target: context-teacher logits
optional target: attention/activation delta matching
```

---

# 7. How to prevent cheating

The trivial cheat is:

> context contains answer → model copies answer → delta encodes answer.

To prevent that, structure the episode so copying is useless.

## 7.1 Support/query separation

Never evaluate on support examples.

```text
support states ≠ query states
```

For cellular automata, support might contain 16 state transitions, but query contains 64 completely new random states under the same rule.

---

## 7.2 Many queries per update

A hard-coded update might help one query, but it will not help 64 unseen queries.

So one update must improve:

```text
Q = {q_1, q_2, ..., q_64}
```

The only viable solution is to infer the rule.

---

## 7.3 OOD query split

Make the query harder than the support.

Example:

```text
support length: 16
query length:   64
```

or:

```text
support: one-step transitions
query:   three-step rollout
```

This forces abstraction.

---

## 7.4 Rule-family holdout

Train on some CA rules and test on unseen CA rules.

Example:

```text
meta-train rules: 0–199
meta-test rules:  200–255
```

Or better, split by structural class rather than numeric ID.

---

## 7.5 Permutation invariance

The update should not depend on accidental support order.

Add:

```text
Uψ(S) ≈ Uψ(shuffle(S))
```

This discourages memorizing sequence position.

---

## 7.6 Update bottleneck

Constrain `Δθ`.

Use:

```text
low-rank ΔW
small norm
few target layers
sparse write gates
```

A small bottleneck forces compression of the rule rather than storage of all examples.

This is analogous to why low-rank/task-vector methods are attractive: in-context learning often appears compressible into compact vectors or low-rank modifications, as seen in task-vector and in-context-vector work. ([arXiv][2])

---

# 8. The architecture I would use

I would separate the system into three parts.

```text
Base model fθ
Context encoder Eψ
Update applicator Aφ
```

## 8.1 Base model

The normal LLM or transformer.

```text
fθ(x)
```

## 8.2 Context encoder

Reads the context/support set:

```text
z = Eψ(S)
```

This is the distilled lesson.

For cellular automata, `z` should encode something like:

```text
the hidden transition rule
```

## 8.3 Update applicator

Turns `z` into a small update:

```text
Δθ = Aφ(z)
```

Possible update forms:

```text
LoRA delta
MLP first-layer delta
attention-bias delta
key/value memory delta
fast-weight delta
```

Then:

```text
f_{θ + Δθ}(x)
```

This is effectively a learned optimizer / learned update rule.

This connects to older meta-learning and learned-optimizer work: MAML trains initial parameters so a few gradient steps generalize, while learned optimizers train update rules themselves; newer ICL work argues transformers can implement implicit learning algorithms inside their activations. ([arXiv][3])

---

# 9. A concrete cellular-automata training batch

One meta-training episode:

```python
rule = sample_rule()
support = sample_transitions(rule, n=16, length=16)
queries = sample_transitions(rule, n=64, length=64)

# 1. raw prediction
p_raw = model(theta, queries.inputs)

# 2. full-context teacher
p_ctx = model(theta, support + queries.inputs)

# 3. predicted persistent update
delta = update_net(support)

# 4. student without context
p_delta = model(theta + delta, queries.inputs)

loss = (
    CE(p_delta, queries.targets)
    + lambda_distill * KL(stopgrad(p_ctx), p_delta)
    + lambda_margin * relu(margin + CE(p_delta, y) - CE(p_raw, y))
    + lambda_retain * KL(model(theta, general_x), model(theta + delta, general_x))
    + lambda_norm * norm(delta)
)
```

The important thing is that `support` is not present in step 4.

The model must carry the rule through `delta`.

---

# 10. The “living organism” version

For a continuous token stream, I would not update permanent model weights every token.

Use three timescales:

## Timescale 1: ordinary context

Fast, temporary.

```text
current context window
```

This is standard ICL.

---

## Timescale 2: fast persistent memory

Medium-term.

```text
session-level adapter / fast weights / memory module
```

Updated every chunk, maybe every 512–4096 tokens.

---

## Timescale 3: slow consolidation

Long-term.

```text
actual base-weight update
```

Updated only after repeated evidence, with replay and retention loss.

This resembles the distinction made in memory-augmented architectures like Titans, where attention acts as short-term memory and a learned neural memory stores longer-term information at test time. ([arXiv][4])

---

# 11. When should the model update?

Use a learned gate:

```text
g_t = sigmoid(Gψ(features_t))
```

Features could include:

```text
loss_raw
loss_with_context
loss_gap = loss_raw - loss_context
surprise
consistency across examples
predicted update norm
agreement between support examples
```

Then:

```text
Δθ_t = g_t · Uψ(C_t)
```

The model should write when:

```text
context helps a lot
examples are mutually consistent
the inferred rule improves held-out predictions
the update does not hurt retention data
```

For cellular automata, the gate should become high once enough examples identify the hidden rule.

For natural language, the gate should become high when context reveals a stable new preference, fact, tool rule, environment rule, or user-specific convention.

---

# 12. The best “attention delta” formulation

If you really want to train an attention-delta object, I would define it as:

```text
Bψ(S, x) = attention-logit bias
```

Then:

```text
Attn_aug(x) = softmax(QK^T / sqrt(d) + Bψ(S, x))
```

Training loss:

```text
L =
  CE(f_aug(x_q), y_q)
+ KL(f_context(S, x_q), f_aug(x_q))
+ β ||Bψ||²
+ γ L_retain
```

But I would not ask the model to output arbitrary attention scores as text or tokens. It should output internal modulation variables.

So instead of:

```text
model produces next_token and attention_score_augmented
```

I would use:

```text
model produces next_token
update head produces Δattention / ΔMLP / ΔLoRA internally
outer loss rewards future improvement
```

The update head can be trained end-to-end.

---

# 13. The most principled version

The strongest design is:

```text
ICL teacher → context-to-update distillation → future-query validation
```

Detailed pipeline:

1. Sample task `τ`.
2. Split examples into support `S`, probe `P`, query `Q`.
3. Run model with context on probes:

   ```text
   fθ(S, p)
   ```

4. Run model without context on probes:

   ```text
   fθ(p)
   ```

5. Compute hidden deltas:

   ```text
   δA_l(S,p) = A_l(S,p) - A_l(p)
   ```

6. Aggregate per-query deltas into a static low-rank update:

   ```text
   ΔW_l*(S)
   ```

7. Train update generator:

   ```text
   Uψ(S) → ΔW_l*(S)
   ```

8. Apply predicted update to model.
9. Evaluate on held-out query `Q` without context.
10. Backprop query loss through the update generator.

That gives you exactly what you asked for:

> train the model to produce meaningful internal deltas that improve future unseen tasks.

---

# 14. Why cellular automata are a good substrate for this

Cellular automata are useful because the hidden rule is compact, but the support examples can be arbitrarily many.

The model must compress:

```text
many observed transitions
```

into:

```text
one reusable rule
```

That is exactly the behavior you want.

A good update does not remember:

```text
state_17 maps to next_state_17
```

It learns:

```text
for each local neighborhood abc, output bit r(abc)
```

Then it applies that rule to new states.

This is a perfect toy environment for testing whether `Δθ` encodes a rule rather than a lookup table.

---

# 15. The final answer

The best way to train this is:

```text
Do not directly optimize next-token accuracy with context.

Instead:

1. Let the model use context to infer a rule.
2. Force it to compress that rule into a small internal update.
3. Remove the context.
4. Evaluate the updated model on new examples.
5. Train the update mechanism only through held-out future performance.
```

The clean objective is:

```text
Δθ = Uψ(S)

minimize:
CE(f_{θ+Δθ}(Q_x), Q_y)
+ KL(fθ(S, Q_x), f_{θ+Δθ}(Q_x))
+ retention_loss
+ update_bottleneck
```

If you want this to be close to the paper, target **MLP/LoRA low-rank updates** derived from:

```text
δA = A(C,x) - A(x)
```

not just raw attention scores.

If you want attention deltas too, use them as internal attention-logit biases and supervise them only indirectly through future-query improvement.

That turns in-context learning into persistent continual learning.

[1]: https://arxiv.org/pdf/2507.16003 "Learning without training: The implicit dynamics of in-context learning"
[2]: https://arxiv.org/abs/2310.15916?utm_source=chatgpt.com "In-Context Learning Creates Task Vectors"
[3]: https://arxiv.org/abs/1703.03400?utm_source=chatgpt.com "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
[4]: https://arxiv.org/abs/2501.00663?utm_source=chatgpt.com "Titans: Learning to Memorize at Test Time"

some ideas on the cellular automata pre-training dataset:

Great—let’s make this concrete.

---

# 🧠 What a cellular automaton actually _does_

A **cellular automaton (CA)** is just:

> A grid of cells that updates over time using a **local rule**

Each cell:

- looks at its **neighbors**
- applies a **fixed rule**
- updates its value

That’s it.

---

## 🔥 Example: 1D Cellular Automaton (simplest)

We use:

- binary cells: `0` or `1`
- a 1D row
- each cell looks at **3 cells**:
  `(left, center, right)`

---

## Rule example: **Rule 110**

This is a famous one (Turing complete).

It defines:

```id="tz6x0u"
Neighborhood → Output

111 → 0
110 → 1
101 → 1
100 → 0
011 → 1
010 → 1
001 → 1
000 → 0
```

---

## ▶️ One step update

### Input:

```id="kebg94"
0 1 1 0 1 0 0
```

Now slide a window of size 3:

```id="m6swt7"
(0 1 1) → 1
(1 1 0) → 1
(1 0 1) → 1
(0 1 0) → 1
(1 0 0) → 0
```

### Output:

```id="g1r4py"
1 1 1 1 0
```

👉 That’s one timestep.

---

# 🧪 Turning this into training data

You want the model to learn:

> “Given a state → predict the next state”

---

## ✅ Training example format (clean)

### Format 1: sequence → next

```id="9sr6nd"
Input:
state: 0 1 1 0 1 0 0

Output:
next:  1 1 1 1 0
```

---

## Format 2: autoregressive (LLM-style)

```id="9xsw9i"
"0 1 1 0 1 0 0 -> 1 1 1 1 0"
```

---

## Format 3: multi-step prediction

```id="h8q5o0"
Input:
t=0: 0 1 1 0 1 0 0

Output:
t=1: 1 1 1 1 0
t=2: 1 0 0 1
```

---

## Format 4: masked rule inference (harder 🔥)

```id="lgdcfj"
Example pairs:
0 1 1 → 1
1 1 0 → 1
0 1 0 → 1

Query:
1 0 0 → ?
```

👉 This forces **rule learning**, not memorization

---

# 🧬 Why this is powerful

The model must learn:

### 1. Local rules

Each output depends only on neighbors

---

### 2. Translation invariance

Same rule applies everywhere

---

### 3. Composition over time

Rules stack:

```id="dy3yd3"
t0 → t1 → t2 → t3
```

---

### 4. Emergence

Simple rules → complex global patterns

---

# 🔥 Best training setup (for you)

If your goal is **learning-to-learn**, use this:

---

## Task A: Next-step prediction

```id="6vwn9w"
Input:  binary sequence
Output: next state
```

---

## Task B: Multi-step rollout

```id="b5b54t"
Input: initial state
Output: state after k steps
```

👉 forces deeper reasoning

---

## Task C: Rule identification (VERY important)

```id="gc31bf"
Given:
examples of transitions

Predict:
new unseen transition
```

👉 this is **meta-learning**

---

## Task D: OOD generalization

Train on:

```id="v2g2h6"
length = 10
```

Test on:

```id="ptz7bd"
length = 50
```

👉 tests **true abstraction**

---

# 🧪 Concrete dataset generator (Python)

Here’s a minimal version:

```python
import random

def apply_rule(state, rule):
    new = []
    for i in range(1, len(state) - 1):
        triplet = (state[i-1], state[i], state[i+1])
        idx = triplet[0]*4 + triplet[1]*2 + triplet[2]
        new.append((rule >> idx) & 1)
    return new

def generate_example(length=10, steps=1, rule=110):
    state = [random.randint(0,1) for _ in range(length)]
    current = state
    for _ in range(steps):
        current = apply_rule(current, rule)
    return state, current

# Example
inp, out = generate_example()
print(inp, "->", out)
```

---

# 🧠 What the model is _actually learning_

Not “patterns” — it’s learning:

### 👉 A function:

```id="gk3tkz"
f(local neighborhood) → next value
```

Then composing it:

```id="86m5wx"
f ∘ f ∘ f ∘ ...
```

---

# 🔥 Why this beats arithmetic (for your goal)

Arithmetic:

- has strong priors
- tied to number system
- partially memorized in LLMs

---

Cellular automata:

- completely synthetic
- no pretraining leakage
- forces **rule discovery from scratch**

---

# 🚀 TL;DR

A cellular automaton training example is:

```id="4n2ndx"
Input:  binary state
Output: next state under hidden rule
```

And the model must learn:

- local rules
- composition over time
- generalization to longer inputs

---

# If you want next step

We can:

- integrate CA into your benchmark
- design a **meta-pretraining curriculum**
- or combine CA + arithmetic to test transfer

Just tell me 👍

I read your post and dug through the benchmark and grokking literature. My overall take is that the post is directionally right, and its strongest idea is also its least common one: continual learning should mean not just _not forgetting_, but _getting better at learning later things because of earlier things_. Classical CL papers usually organize the space into task-incremental, domain-incremental, and class-incremental settings; your arithmetic proposal is closest to task-incremental learning with explicit OOD rule generalization and positive transfer, which is a stronger requirement than many standard protocols actually enforce. ([Nature][1])

What the post needs most is a sharper evaluation contract. “Preserve general capability” has to become an explicit held-out suite, and the field’s standard reporting should be named directly: average accuracy, backward transfer, and forward transfer, ideally alongside model-size, replay-storage, and compute efficiency. It also helps to separate **skill acquisition** from **temporal knowledge updating**. TiC-LM, for example, is about updating an LLM over 114 months of Common Crawl; your arithmetic program is about learning reusable operators and then composing them. TiC-LM is also a concrete warning for your “overwriting knowledge” section: replay helped stable knowledge like NumPy but hurt fast-changing knowledge like PyTorch. ([arXiv][2])

The harness-versus-parametric split in the post is useful, but a bit too binary. Recent benchmarks such as MemoryBench and LifelongAgentBench treat system memory itself as a continual-learning problem, not a mere workaround, so a rigorous research agenda probably ought to compare in-weight learning, external memory, and hybrids under the same retention/transfer budget rather than assuming only one of them counts as “real” CL. ([OpenReview][3])

On benchmarks, the historically popular CL suite is still dominated by vision. Split MNIST and Split CIFAR-100 remain canonical, while Permuted-MNIST is older and increasingly criticized as artificial. CORe50 was proposed as a continuous object-recognition benchmark, CLEAR was introduced specifically to replace artificial temporal variation with real-world imagery evolving from 2004 to 2014, and Stream-51 pushes toward temporally correlated video streams plus novelty detection. Those are the benchmarks people still mean when they say “popular continual-learning benchmarks,” even if they are not what you ultimately want. ([Nature][1])

For LLMs, the benchmark ecosystem is much newer. TRACE uses eight datasets spanning domain tasks, multilingual ability, code generation, and math reasoning, and its abstract reports that aligned models can lose both general ability and instruction following after sequential training; in the paper’s headline example, Llama-2-chat-13B on GSM8K drops from 28.8% to 2% after the TRACE sequence. TiC-LM targets time-continual pretraining over 114 monthly Common Crawl slices with time-stratified evaluations such as Wikipedia, StackExchange, and code docs. At the system level, MemoryBench evaluates learning from accumulated user feedback, while LifelongAgentBench tests lifelong skill accumulation for agents in database, OS, and knowledge-graph environments. ([arXiv][4])

The benchmark closest to what you want already exists in partial form: **CLeAR** (NeurIPS 2023), a continual algorithmic-reasoning benchmark covering 15 tasks over formal-language and arithmetic-style concepts. It matters because it explicitly argues that algorithmic-reasoning CL is different from image CL: the goal is OOD rule generalization, not IID distribution matching, and replay samples like a single arithmetic fact are poor representatives of the underlying concept. CLeAR also makes knowledge transfer part of the objective, not just remembering. ([NeurIPS Proceedings][5])

On grokking: the original grokking paper is exactly about the issue you care about at a mechanistic level. On small algorithmic datasets, models can fit the training data, appear overfit, and only much later jump to perfect generalization; smaller datasets often require _more_ optimization before that jump happens. That makes grokking a very good microscope for the question “did the model learn the rule or memorize the examples?” even though, by itself, it is not yet a continual-learning setup. ([arXiv][6])

The most important grokking result for your purposes is that the field is converging on **structure formation**, not mere rote fitting, as the key story. Mechanistic-interpretability work on modular addition found that small transformers can implement a discrete-Fourier / trigonometric circuit. Gromov showed that even simple two-layer networks can grok modular arithmetic and learn analyzable task-specific feature maps. More recent AGOP/RFM work goes further and argues that grokking is not unique to neural nets or SGD at all: it can arise from gradual feature learning in non-neural kernel systems too. ([arXiv][7])

That said, there is no single settled explanation. Omnigrok argues for an “LU” mismatch between train and test loss versus weight norm and shows grokking can be induced beyond algorithmic data; a margin-maximization account explains modular-arithmetic grokking as gradient descent eventually leaving an early kernel-like regime; lottery-ticket work ties the delay to discovery of good subnetworks; and Grokfast shows that amplifying slow gradients can reduce the delay by more than 50×. A very recent 2026 preprint on multi-task modular arithmetic suggests shared-trunk models can grok several arithmetic tasks together in staggered order, which is highly relevant to your sequential-skill idea, but I would treat that result as promising rather than settled. ([arXiv][8])

So is arithmetic a good task for continual learning? Yes — in fact, it is one of the best **fruit-fly benchmarks** available, provided you use it correctly. It gives you infinite procedurally generated data, exact correctness, strong OOD splits by length/value/base/surface form, and unusually good visibility into what the model learned. Both CLeAR and the newer AttentionSpan work push in this direction, arguing for configurable generators precisely because fixed test sets blur extrapolation, memorization, and contamination. ([NeurIPS Proceedings][5])

But arithmetic is not enough to stand in for continual learning in LLMs as a whole. If the operator token is provided, the task is usually task-incremental rather than the harder class-incremental case; the symbols are clean, the distributions are controlled, and the failure modes can become very algebra-specific. So I would treat arithmetic as a microscope for transfer, composition, and mechanism formation — not as the whole evaluation suite. ([Nature][1])

The strongest version of your experiment would look like this. First, use **infinite generators**, not fixed corpora. Hold out not just examples, but entire difficulty axes: longer operand lengths, new value ranges, new numeral bases, token remappings, and new prompt templates. That is the cleanest way to block contamination and force rule learning. Second, measure transfer directly using the full task-by-time accuracy matrix plus BWT/FWT, but add the metric you actually care about: how many examples or updates are needed to hit a fixed OOD threshold on the next task, compared with a from-scratch baseline. That “examples-to-threshold” number is the clearest operationalization of your claim that the model is getting better at learning. Third, include a genuine composition task, not just another nearby task — for example: T1 addition, T2 multiplication, T3 mixed arithmetic expressions or affine forms that require both earlier skills. The key score is zero-shot or few-shot performance on T3 _before_ direct training, and then adaptation speed once a little T3 data is allowed. Fourth, run both a **task-ID given** and **task-ID hidden** version, so you can separate pure transfer from rule-family inference. ([arXiv][9])

A richer curriculum would probably start even earlier than addition. CLeAR’s framing suggests beginning with primitives such as parity-like checks, carry-relevant dependencies, value assignment, Dyck-style nesting, or other formal-language tasks, then moving to addition, multiplication, and finally composed programs. That gives you a way to test whether earlier “micro-skills” reduce the sample complexity of later arithmetic tasks. ([NeurIPS Proceedings][5])

I would also add mechanistic readouts, not just accuracy. AttentionSpan gives one route via reference attention masks; the grokking literature gives others via Fourier-style circuit probes or AGOP-style feature analysis. If the model truly learns reusable first principles, you want to see shared internal structure emerge before or alongside faster adaptation, not just a black-box accuracy bump. ([arXiv][9])

On your meta-pretraining question — what abstract substrate should be used if the downstream continual tasks are arithmetic? — the best current evidence points to **structured synthetic data with long-range dependencies**, not bare random byte or bit operations. Formal-language pre-pretraining can improve later language acquisition when the formal task matches the model’s computational biases. Procedural pretraining on Dyck and related generators has reported large gains in context recall and better data efficiency, and follow-up work argues that different procedural rules plant complementary inductive structures in different parts of the network. ([arXiv][10])

Formal logic is another especially strong candidate. FLD was designed to isolate deduction from world knowledge and reported that even GPT-4 solved only about half the problems; ALT on the newer FLD×2 corpus reported large gains on logical reasoning and smaller transfer gains on math, coding, and BBH. If your goal is “learn abstract inference rules without leaking the downstream arithmetic tasks,” logic-style synthetic corpora are one of the cleanest options available. ([arXiv][11])

Cellular automata are the most interesting speculative option. One paper shows transformers can abstract elementary cellular-automata rules for next-step prediction, although multi-step planning remains much harder. Another very recent paper reports that pre-pretraining on neural cellular automata can improve downstream language modeling, transfer to reasoning benchmarks, and speed convergence, with only a relatively small synthetic corpus. My read is that CA-like data are promising because they are controllable, scalable, and structurally rich — but the evidence here is newer and less settled than for formal languages or logic. ([arXiv][12])

So my bottom line is: your post is conceptually strong, especially in elevating **compositional positive transfer** to a first-class criterion, but it should formalize itself around scenario type, OOD evaluation, BWT/FWT, and transfer-efficiency curves. Arithmetic and grokking are not side issues here; they are probably the cleanest laboratory we have for testing whether a model learns reusable principles rather than episodic facts. The benchmark I would trust most is a hybrid of **CLeAR** and **AttentionSpan**, with modern LLM-style reporting borrowed from **TRACE** and **TiC-LM**, and a meta-pretraining phase built from **formal languages + formal logic + possibly cellular automata**, while keeping downstream arithmetic completely disjoint from the pretraining generators. ([NeurIPS Proceedings][5])

The next useful step is to turn this into an explicit benchmark spec: task generators, OOD splits, metrics, and a baseline matrix of replay / regularization / adapter / full-update methods.

[1]: https://www.nature.com/articles/s42256-022-00568-3 "https://www.nature.com/articles/s42256-022-00568-3"
[2]: https://arxiv.org/pdf/1810.13166 "https://arxiv.org/pdf/1810.13166"
[3]: https://openreview.net/forum?id=wU4Tjlzg3h "https://openreview.net/forum?id=wU4Tjlzg3h"
[4]: https://arxiv.org/abs/2310.06762 "https://arxiv.org/abs/2310.06762"
[5]: https://proceedings.neurips.cc/paper_files/paper/2023/file/986e0caad271b59417287737416d8594-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2023/file/986e0caad271b59417287737416d8594-Paper-Conference.pdf"
[6]: https://arxiv.org/abs/2201.02177 "https://arxiv.org/abs/2201.02177"
[7]: https://arxiv.org/abs/2301.05217 "https://arxiv.org/abs/2301.05217"
[8]: https://arxiv.org/abs/2210.01117 "https://arxiv.org/abs/2210.01117"
[9]: https://arxiv.org/html/2503.01909v1 "https://arxiv.org/html/2503.01909v1"
[10]: https://arxiv.org/abs/2502.19249 "https://arxiv.org/abs/2502.19249"
[11]: https://arxiv.org/abs/2308.07336 "[2308.07336] Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic"
[12]: https://arxiv.org/abs/2412.01417 "https://arxiv.org/abs/2412.01417"
