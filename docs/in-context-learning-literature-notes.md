# In-Context Learning, Inner Optimization, and Consolidation

Generated on 2026-03-21.

This note starts from the seed papers we already discussed, then expands outward one to two citation hops into the papers that are most relevant to the specific agenda here:

- What mechanism actually powers in-context learning (ICL)?
- When does the model generalize instead of memorize?
- Can transient prompt learning be converted into persistent weights or memory?
- What architectures or objectives make the inner learner better?
- How does all of this connect to learned optimization and continual adaptation?

This is not an exhaustive bibtex dump. It is a curated recursive reading note around the mechanism question.

## High-Level Map

```text
                  +----------------------+
                  |  Few-shot behavior   |
                  |  in large LMs        |
                  +----------+-----------+
                             |
                             v
         +-------------------+-------------------+
         |                                       |
         v                                       v
+---------------------+              +------------------------+
| Mechanism papers    |              | Generalization papers  |
| Bayes / GD / heads  |              | phases / grokking      |
+----------+----------+              +-----------+------------+
           |                                         |
           v                                         v
+----------+----------+              +---------------+--------+
| Consolidation       |              | Adaptive compute /     |
| context -> weights  |              | test-time learning     |
+----------+----------+              +---------------+--------+
           |                                         |
           +-------------------+---------------------+
                               |
                               v
                    +----------+----------+
                    | Better inner        |
                    | learners and        |
                    | outer objectives    |
                    +---------------------+
```

## 1. Foundations and Mechanistic Theories

### 1. Brown et al. (2020) - Language Models are Few-Shot Learners

Primary link: [arXiv](https://arxiv.org/abs/2005.14165)

This is the empirical starting gun. The paper does not explain ICL mechanistically, but it establishes the phenomenon cleanly enough that the rest of the literature exists. The important point is not merely that GPT-3 performs better with examples in the prompt; it is that scaling produces a model that can often infer a task from demonstrations without any gradient updates at inference time. Brown et al. therefore define the outer behavioral target: prompts can act like temporary task descriptions, and larger models become increasingly able to condition on them. For this reading list, the paper matters because it frames the central puzzle: why does plain next-token prediction create a model that sometimes behaves as if it is fitting a new predictor on the fly?

```text
pretraining on text
       |
       v
  large causal LM
       |
       v
[demo 1][demo 2]...[query]
       |
       v
   task-conditioned answer
```

### 2. Xie et al. (2021) - An Explanation of In-context Learning as Implicit Bayesian Inference

Primary link: [arXiv](https://arxiv.org/abs/2111.02080)

Xie et al. give one of the cleanest non-optimization stories for ICL: the model learns during pretraining to infer latent document-level structure, and prompt-time ICL reuses that latent inference machinery. In their synthetic mixture-of-HMM setting, the transformer is not literally running SGD in context; it is inferring a shared hidden concept that explains the examples and then using that latent to predict the query. This matters because it sets up a durable tension in the literature: some later papers interpret ICL as implicit optimization, while this paper shows that at least part of the story can be cast as latent-variable inference over coherent generative structure. For your agenda, this paper is a strong reminder that "learning in context" may sometimes be better understood as fast posterior inference than as token-space gradient descent.

```text
examples in prompt
   (x1,y1) (x2,y2) ... (xk,yk)
            |
            v
   infer shared latent concept z
            |
            v
       answer query x*
```

### 3. Elhage et al. (2021) - A Mathematical Framework for Transformer Circuits

Primary link: [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

This paper is not an ICL paper in the narrow sense, but it gives the circuit language that later mechanistic ICL work relies on. Its main contribution is to decompose attention-only transformers into interpretable path compositions, especially query-key and output-value circuits, and to show how multi-layer attention can implement specific algorithms. The key relevance here is that the paper introduces induction heads as a concrete, reverse-engineered motif rather than as vague intuition. If you care about whether the model is doing "inner learning" versus mere retrieval, this framework is useful because it shows how concrete algorithms can be embedded in the forward pass. It also pushes the field toward a programmatic stance: instead of only measuring prompt performance, inspect the actual circuit that carries the prompt information forward.

```text
token stream
    |
    v
[QK circuit] -> decides where to look
    |
    v
[OV circuit] -> decides what to copy/write
    |
    v
composed multi-head algorithm
```

### 4. Olsson et al. (2022) - In-context Learning and Induction Heads

Primary link: [Transformer Circuits / arXiv mirror](https://arxiv.org/abs/2209.11895)

Olsson et al. are the canonical "induction heads explain ICL" paper. The core claim is that a specific attention-head motif learns to continue repeated patterns such as `[A][B] ... [A] -> [B]`, and that the emergence of this circuit coincides with a sharp training transition in small transformers. The paper is strongest when interpreted narrowly: induction heads clearly explain a lot of sequence continuation behavior and likely some of the generic copy-and-continue substrate that supports broader ICL. The more controversial leap is treating induction heads as the main mechanism for most ICL in large models. For your purposes, the important takeaway is that there is at least one crisp, mechanistically identifiable forward-pass circuit that gives models a "learn from previous examples in this prompt" capability without weight updates.

```text
... A  B  ...  A  ?
        ^      |
        |      |
   previous B <- match current A to earlier A

result: predict B
```

### 5. Garg et al. (2022/2024) - What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

Primary link: [arXiv](https://arxiv.org/abs/2208.01066)

Garg et al. ask a very useful question: if we strip away messy natural language and just train transformers from scratch to do ICL on clean function families, what function classes can they actually learn? Their answer is stronger than many people expected. Standard transformers can learn unseen linear functions in context and can also handle more structured families such as sparse linear functions, small neural nets, and decision trees. The paper matters because it converts "ICL is mysterious" into a more measurable question about the hypothesis classes that are learnable through prompt-time adaptation. It also shows that ICL is not confined to memorizing training tasks; with the right setup, transformers really can learn reusable estimators over task families. This makes the later mechanistic papers more credible, because there is a clear synthetic benchmark where the model is genuinely doing task inference rather than merely parroting format.

```text
sample task f ~ task family F
      |
      v
prompt: (x1,f(x1))...(xk,f(xk)), x*
      |
      v
 transformer learns estimator for F
      |
      v
 predict f(x*)
```

### 6. Akyurek et al. (2022/2023) - What Learning Algorithm Is In-Context Learning? Investigations with Linear Models

Primary link: [arXiv](https://arxiv.org/abs/2211.15661)

Akyurek et al. are one of the central bridges from behavior to algorithm. They focus on linear regression and show, by both construction and experiment, that transformers can implement familiar estimators such as gradient descent, ridge regression, least squares, and in large limits even Bayesian estimators. The important conceptual move is not just "transformers can fit linear models," but "the forward pass can contain an implicit small learner whose sufficient statistics are encoded in activations." The paper is therefore much closer to your intuition that the prompt gets translated into a latent task representation that behaves like a learned parameter state. It also points to a useful research stance: stop asking whether the transformer learns and start asking which learning algorithm it has implicitly discovered, under what depth/width/data regimes.

```text
examples -> hidden state accumulates:
           w_t, X^T X, X^T y
                     |
                     v
        implicit solver in late layers
                     |
                     v
               prediction for x*
```

### 7. Kirsch et al. (2022/2024) - General-Purpose In-Context Learning by Meta-Learning Transformers

Primary link: [arXiv](https://arxiv.org/abs/2212.04458)

Kirsch et al. push the more ambitious meta-learning view: black-box models can be meta-trained to act as fairly general in-context learners, even without hand-specifying a particular inner optimizer or inference rule. The result that matters most here is their phase picture: depending on model size, task count, and optimization, the system can fall into memorization, genuine generalization, or outright meta-training failure. They also argue that the effective bottleneck is not just parameter count but accessible state size, meaning the size and structure of the internal memory available for the next prediction. This is directly relevant to your interest in "better ways to learn." If the inner learner is state-bottlenecked, then architectural choices about recurrence, looping, or memory may matter as much as raw scale.

```text
outer loop: sample many tasks
            |
            v
    meta-train transformer
            |
            v
inner loop happens in activations
            |
            +--> memorize
            +--> generalize
            +--> fail to meta-learn
```

### 8. von Oswald et al. (2022/2023) - Transformers Learn In-Context by Gradient Descent

Primary link: [arXiv](https://arxiv.org/abs/2212.07677)
Code: [official notebooks](https://github.com/transformerGD/transformers-learn-in-context-by-gradient-descent)

This is the flagship "ICL as inner gradient descent" paper. The paper first gives a constructive equivalence between a single linear self-attention layer and one gradient-descent step on a regression objective, then shows empirically that trained transformers on synthetic tasks converge toward that construction. What makes the paper influential is that it turns a fuzzy analogy into a real mechanistic program: if the forward pass is learning, then maybe it is literally implementing an optimizer. It also goes beyond first-order matching by noting that transformers can beat plain GD by learning curvature corrections and richer representations. For your agenda, this paper is central because it suggests a route to designing better ICL systems: modify the architecture or training setup so that the learned inner optimizer is more stable, more expressive, or better aligned with generalization.

```text
prompt examples
      |
      v
self-attention layer
      |
      +--> equivalent to one GD-like update
      |
      v
new latent predictor state
      |
      v
query prediction
```

### 9. Dai et al. (2023) - Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers

Primary link: [ACL Anthology](https://aclanthology.org/2023.findings-acl.247/)
Code: [project link](https://aka.ms/icl)

Dai et al. take the optimization story from synthetic regression toward pretrained language models. Their framing is that ICL behaves like implicit finetuning: the demonstrations produce "meta-gradients," and the transformer uses attention to apply the effect of those gradients without changing persistent parameters. The paper is valuable because it compares ICL with explicit finetuning behaviorally on real tasks, rather than only proving algorithmic equivalence in stylized settings. It also proposes momentum-based attention inspired by optimizer design, which is important for your direction: if we really believe forward-pass learning is optimizer-like, then optimizer design ideas should transfer back into architecture. This is one of the strongest papers supporting the thought that a model could be trained not only to solve tasks in context, but to prefer better update trajectories.

```text
demos
  |
  v
attention computes meta-gradient signal
  |
  v
implicit "finetuned" model state
  |
  v
prediction on query

attention ~ optimizer step
```

### 10. von Oswald et al. (2023/2024) - Uncovering Mesa-Optimization Algorithms in Transformers

Primary link: [arXiv](https://arxiv.org/abs/2309.05858)

This paper makes the inner-optimizer hypothesis even more concrete by reverse-engineering synthetic autoregressive transformers and explicitly identifying subsidiary optimization algorithms in the forward pass. The key advance over the earlier GD-equivalence line is that the authors do not merely show a possible equivalence; they argue that next-token training itself can induce a principled internal objective and gradient-based algorithm that adjusts predictions as more input arrives. In other words, the model is not only representing task information; it is running an inner learning process because that is useful for the outer autoregressive objective. This is highly relevant to your idea of a learned "taste" over updates. Once there is an inner algorithm, one can ask what objective it optimizes, what inductive biases it inherits, and whether we can train it to favor trajectories that generalize instead of merely fitting the immediate prompt.

```text
outer objective: next-token loss
            |
            v
 learned forward-pass algorithm
            |
            v
 inner objective over current sequence
            |
            v
 updated predictions as tokens arrive
```

## 2. Theory of Generalization, Phase Changes, and Representations

### 11. Hendel et al. (2023) - In-Context Learning Creates Task Vectors

Primary link: [arXiv](https://arxiv.org/abs/2310.15916)

Hendel et al. propose a very clean compression view: instead of thinking of the model as carrying all demonstrations forward token by token, think of it as compressing the support set into a single latent task vector that modulates prediction on the query. This is extremely relevant to your "task factor" intuition. The paper does not say that the full prompt is irrelevant, but it argues that in many settings the effective computation can be reduced to `theta(S)`, a compact representation of the support set. This idea is useful both mechanistically and architecturally. Mechanistically, it gives you an object to try to probe or regularize. Architecturally, it suggests explicit fast-weight or memory-slot designs where the prompt is summarized into a reusable latent. It also creates a bridge to consolidation: if prompts naturally collapse into task vectors, perhaps those vectors are what should be replayed, clustered, or distilled into persistent memory.

```text
support set S = {(x,y)}
        |
        v
compress S -> task vector theta(S)
        |
        v
model(query, theta(S)) -> answer
```

### 12. Fu et al. (2024) - Transformers Learn to Achieve Second-Order Convergence Rates for In-Context Linear Regression

Primary link: [arXiv](https://arxiv.org/abs/2310.17086)

Fu et al. are important because they challenge the oversimplified "ICL is just gradient descent" story. In their linear-regression setting, trained transformers align much more closely with iterative Newton-style updates than with first-order GD, including behavior on ill-conditioned problems. That is a major result for your agenda, because it suggests the forward-pass learner can discover not just any optimizer, but one with sophisticated curvature handling. This matters if you want a model to prefer better internal update directions. The paper implies that the inner learner can already discover high-quality algorithms when the training distribution rewards them. It also strengthens the case for recurrent or looped computation: if layers correspond to optimizer iterations, then extra reusable depth is a direct way to buy more inner optimization steps.

```text
layer 1  -> approx Newton iter 1
layer 2  -> approx Newton iter 2
layer 3  -> approx Newton iter 3
 ...
final prediction

deeper stack => more inner optimization
```

### 13. Yang et al. (2024) - In-Context Learning with Representations: Contextual Generalization of Trained Transformers

Primary link: [arXiv](https://arxiv.org/abs/2408.10147)

Yang et al. ask a subtler question than simple linear-regression papers: can a transformer learn contextual generalization when the prompt only partially identifies a latent template and labels are noisy? Their answer is yes, under a representation-learning view. In the analyzed setup, the transformer effectively learns to perform ridge regression over basis functions that span the task family. This paper matters because it moves beyond toy "fit the obvious linear rule" benchmarks toward underdetermined contexts where the model must infer a structured latent representation of the task. That is closer to your concern that good ICL should not merely fit examples but discover more general principles behind them. The paper suggests that representation learning and inner estimation are intertwined: the prompt learner is more powerful when it can infer the right basis in which task structure becomes simple.

```text
task template = sum_i a_i phi_i(x)
            |
prompt gives noisy partial labels
            |
            v
 transformer infers coefficients a_i
            |
            v
 contextual generalization on x*
```

### 14. Lu et al. (2024/2025) - Asymptotic Theory of In-Context Learning by Linear Attention

Primary link: [arXiv](https://arxiv.org/abs/2405.11751)

Lu et al. provide one of the clearest phase-transition stories in a solvable setting. In their linear-attention analysis of in-context linear regression, the key control variable is task diversity. Low diversity yields memorization of training tasks; high diversity yields genuine ICL that generalizes outside the pretrained tasks. This is very close to your concern that "just crystallizing a task into weights" is not enough; what we want is abstraction pressure. The paper says that abstraction can emerge when the training distribution forces it, because memorizing many tasks becomes worse than learning the family structure. For research design, this paper is useful because it turns a vague intuition into a measurable outer-loop variable: if you want better inner learners, control the diversity and compositionality of the task distribution they are meta-trained on.

```text
task diversity low  -> memorize seen tasks
task diversity high -> infer task family rule

pretraining task diversity
           |
           v
   phase transition in ICL behavior
```

### 15. Nguyen and Reddy (2024) - Differential Learning Kinetics Govern the Transition from Memorization to Generalization During In-Context Learning

Primary link: [arXiv](https://arxiv.org/abs/2412.00104)

Nguyen and Reddy sharpen the phase-transition picture by arguing that memorization and generalization are not simply one circuit replacing another because of capacity limits. Instead, the two sub-circuits can be largely independent, and the observed transition arises from differential learning speeds. This is an important conceptual correction. If true, then "encourage generalization" does not necessarily require reducing memorization capacity; it may require changing which sub-circuit wins the training race. That is directly relevant to your idea of preferring some update paths over others. The paper suggests that the outer training objective could shape learning kinetics so that the more abstract, transferable algorithm forms earlier or dominates more reliably. It also helps explain transient ICL behavior near the threshold, where different algorithms can coexist and compete.

```text
two sub-circuits emerge in training:

memorizer --------\
                   > compete over behavior
generalizer ------/

winner depends on relative learning speed
```

### 16. Park et al. (2024/2025) - Competition Dynamics Shape Algorithmic Phases of In-Context Learning

Primary link: [arXiv](https://arxiv.org/abs/2412.01003)

Park et al. push the "mixture of algorithms" viewpoint. On their sequence-modeling benchmark, model behavior can be decomposed into several broad strategies that blend retrieval versus inference with unigram versus bigram statistics. Changes in context size or training amount shift which algorithm dominates. This paper is especially relevant if you dislike monolithic stories of ICL. It suggests there may be no single answer to "how does ICL work?" because models can switch among multiple internal solvers depending on regime. For your agenda, this is useful and slightly sobering. If you want to train a model to prefer elegant, generalizing inner updates, you may need an outer controller or value-like signal that selects among multiple available algorithms instead of assuming a single inner optimizer will naturally dominate.

```text
available internal strategies:

retrieval + unigram
retrieval + bigram
inference + unigram
inference + bigram

training/context regime picks the winner
```

### 17. Ren et al. (2024) - Identifying Semantic Induction Heads to Understand In-Context Learning

Primary link: [arXiv](https://arxiv.org/abs/2402.13055)

Ren et al. extend the induction-head story from surface repeated-token copying toward semantically structured relations. Their claim is that some attention heads act as semantic induction heads: when they attend to head entities or syntactic governors, they retrieve semantically related tail content and boost corresponding logits. The paper matters because it suggests that the induction story scales from exact string continuation toward more abstract relational recall. For your interests, this is a partial bridge between bare pattern-matching and principled task inference. It does not yet solve the problem of abstract rule formation, but it shows that some prompt-time adaptation may proceed by retrieving semantically structured relations rather than raw token n-grams. That makes induction heads a richer substrate than the simplest copy-circuit interpretation implies.

```text
head token/entity
      |
      v
semantic induction head
      |
      v
retrieve related tail / relation target
      |
      v
boost matching output tokens
```

### 18. Gatmiry et al. (2024) - On the Role of Depth and Looping for In-Context Learning with Task Diversity

Primary link: [arXiv](https://arxiv.org/abs/2410.21698)

This paper is one of the best direct matches to your recursive-compute questions. Gatmiry et al. show that for diverse in-context regression tasks, depth is not incidental: you need enough layers to handle the condition-number range of the task family. But they also show that plain deep transformers buy expressivity at the cost of fragility, while looped transformers with weight sharing can retain expressive power and gain robustness. That is a very important result if you want the model to iteratively refine its latent hypothesis. It suggests that reusable computation, not just fixed-depth stacking, may be the right way to increase inner optimization depth. In short, this is a strong theoretical endorsement of "run the same learning rule for more steps" architectures rather than only scaling the stack.

```text
standard deep transformer:
  layer1 -> layer2 -> ... -> layerL

looped transformer:
  same block B applied repeatedly

context -> B -> B -> B -> ... -> answer
           ^ shared weights ^
```

## 3. Grokking and the Difference Between Fitting and Rule Discovery

### 19. Power et al. (2022) - Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets

Primary link: [arXiv](https://arxiv.org/abs/2201.02177)

Power et al. are not writing about ICL, but the paper is foundational for the exact worry you raised: there is a big difference between fitting a task and discovering the underlying algorithmic principle. Grokking is the phenomenon where a model memorizes first, then after prolonged training suddenly generalizes. This is useful for ICL research because it provides a vocabulary for delayed abstraction. If an inner learner can also memorize the prompt before discovering the right invariant, then we should not evaluate it only by short-horizon correctness. We should also ask whether the system is heading toward a more compressible, symmetric, rule-like representation. In that sense, grokking is a prototype for the distinction between low-quality and high-quality internal updates that you want to operationalize.

```text
training time ----------------------------->

train accuracy:  high early
test accuracy:   low -----> sudden jump

memorize first, generalize later
```

### 20. Wang et al. (2024) - Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization

Primary link: [arXiv](https://arxiv.org/abs/2405.15071)
Code: [official repo](https://github.com/OSU-NLP-Group/GrokkedTransformer)

Wang et al. connect grokking directly to transformer reasoning circuits. Their central result is that transformers can learn implicit reasoning over parametric knowledge, but often only through a grokking phase where generalizing circuits eventually overtake memorizing ones. The paper is particularly relevant to your desire for more principled internal updates because it studies the competition between memorization circuits and generalizing circuits mechanistically, not just behaviorally. It also makes an important point about systematicity: different reasoning types generalize differently, and the internal circuit configuration partly explains why. For your agenda, this suggests a concrete research program: inspect and score intermediate circuits, not just final answers, and train the model so that the circuit geometry associated with reusable reasoning forms earlier and more reliably.

```text
memorization circuit ----\
                          > training competition -> final behavior
generalizing circuit ----/

late in training:
generalizing circuit dominates
```

## 4. Consolidating Prompt Knowledge into Persistent State

### 21. Snell et al. (2022) - Learning by Distilling Context

Primary link: [arXiv](https://arxiv.org/abs/2209.15189)

Snell et al. are one of the clearest answers to your "why does the model have to relearn every time?" complaint. Their idea is to use context distillation: if instructions, scratchpads, or prompts help the model solve a task, then train the same model to reproduce the improved final answer without the extra context. In other words, treat prompt-time competence as a teacher and absorb it into parameters. The paper is important because it operationalizes internalization without requiring labeled data beyond the model's own contextualized outputs. Its limitation, which matters for your agenda, is that it mainly distills performance, not necessarily abstract principle quality. Distillation can crystallize useful behavior, but unless the teacher itself has found a general rule, the student may just inherit a shallow shortcut. That is exactly where your proposed emphasis on reflection or update quality would need to intervene.

```text
[instructions + scratchpad + input]
               |
               v
         teacher output
               |
      distill final answer only
               |
               v
     student(input) -> answer
```

### 22. Chen et al. (2024) - Exact Conversion of In-Context Learning to Model Weights in Linearized-Attention Transformers

Primary link: [arXiv](https://arxiv.org/abs/2406.02847)

Chen et al. make the prompt-to-weights bridge explicit. In linearized-attention transformers, they show that the effect of demonstrations can be converted exactly into bias-like parameter modifications, and approximately in more standard transformers. This is one of the strongest formal statements behind your idea that a task vector in context might be transformed into a parameter update that persists beyond the current conversation. The paper is especially useful because it does not treat prompting and finetuning as categorically separate operations; it treats them as different parameterizations of related computations. The caveat is that the strongest exact result holds in a simplified architecture. Still, the paper gives a concrete algorithmic path for "crystallizing" ICL into weights and is therefore a key stepping stone toward systems that alternate between ephemeral prompt learning and selective long-term consolidation.

```text
prompt with demos
      |
      v
ICL effect on activations
      |
 exact/approx conversion
      |
      v
equivalent bias / weight patch
      |
      v
same behavior without prompt
```

### 23. Dherin et al. (2025) - Learning Without Training: The Implicit Dynamics of In-Context Learning

Primary link: [arXiv](https://arxiv.org/abs/2507.16003)

Dherin et al. give a strikingly direct account of ICL as implicit fast-weight writing. Their key claim is that a transformer block with self-attention followed by an MLP can be understood as using context to induce a low-rank update to the MLP weights inside the forward pass. This is important because it turns an analogy into an internal dynamical mechanism: the model is not merely storing the prompt in activations; it is functionally writing a temporary weight patch. For your research direction, this is one of the most relevant papers in the entire list. It suggests that the right abstraction may be a two-timescale learner where prompt-time computation already behaves like local parameter editing. If that is true, then better ICL may require better temporary update rules and better criteria for deciding which temporary updates deserve promotion into persistent memory.

```text
context tokens
      |
      v
self-attention computes patch signal
      |
      v
MLP behaves as if:

W_mlp  ->  W_mlp + DeltaW(context)

temporary fast weights at inference time
```

### 24. Mazzawi et al. (2025/2026) - Transmuting Prompts into Weights

Primary link: [arXiv](https://arxiv.org/abs/2510.08734)

Mazzawi et al. build on the implicit-fast-weight view and generalize it from token-dependent local patches to token-independent "thought vectors" and "thought matrices" that summarize prompt information into reusable control objects. This is unusually close to your idea that the system should distill not just surface examples but a more generalized latent principle. The paper's big contribution is to unify activation steering and weight editing under one transformer-grounded story: prompt information can be re-expressed as reusable vectors or matrices that shape model behavior. That makes it a bridge between prompting, interpretability, and model editing. For your agenda, the important implication is that a prompt may already contain a compressible internal object richer than a single answer trace, and that training could be designed to favor those compressed objects when they transfer better than raw demonstrations.

```text
prompt chunk
    |
    v
token-dependent internal updates
    |
compress / condense
    |
    +--> thought vector
    +--> thought matrix
    |
    v
reusable control / weight update object
```

### 25. Goldwaser et al. (2025) - Equivalence of Context and Parameter Updates in Modern Transformer Blocks

Primary link: [arXiv](https://arxiv.org/abs/2511.17864)

Goldwaser et al. extend the prompt-to-weights story from simpler blocks to modern architectures such as Gemma-style blocks, RMSNorm, gating, MoE, and parallel or sequential block variants. The key insight is that the equivalence between context effects and weight updates is not a weird artifact of one toy block, but may be a fairly broad structural fact when certain controllability conditions hold. This matters because it upgrades the earlier prompt-to-weight arguments from interesting toy results to something that may genuinely apply to current LLM design. For your research direction, the paper is important because it suggests that "ephemeral learning as implicit weight writing" may be a property of modern transformer blocks in general. If so, then outer-loop research should explicitly reason about which temporary updates are high quality, general, and worth stabilizing.

```text
modern transformer block
   attention + norm + gated MLP + ...
                |
                v
context effect <=> rank-1 / low-rank parameter patch
                |
                v
prompting and local editing become formally linked
```

## 5. Adaptive Learning at Test Time and Long-Term Memory

### 26. Hardt and Sun (2024) - Test-Time Training on Nearest Neighbors for Large Language Models

Primary link: [ICLR 2024 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/f02f1185b97518ab5bd7ebde466992d3-Paper-Conference.pdf)
Code: [official repo](https://github.com/socialfoundations/tttlm)

Hardt and Sun offer a practical hybrid answer to the persistence problem: retrieve nearest-neighbor text, finetune briefly at test time on those retrieved examples, then answer. This is not pure ICL and not full continual training either; it is a test-time adaptation scheme that uses retrieval to construct a tiny task-specific training set. The paper matters because it concretely separates three things that are often conflated: stored external memory, local parameter updates, and final generation. For your agenda, this is useful as a systems baseline. If you want a model to consolidate or refine principles online, one simple route is to use retrieval to stage candidate evidence and then perform a small local update. The weakness, of course, is that nearest-neighbor adaptation tends to be opportunistic rather than principle-seeking unless the retrieval set itself is curated for abstraction.

```text
query
  |
  v
retrieve nearest neighbor text
  |
  v
small test-time finetuning step
  |
  v
adapted LM answers query
```

### 27. Sun et al. (2024/2025) - Learning to (Learn at Test Time): RNNs with Expressive Hidden States

Primary link: [arXiv](https://arxiv.org/abs/2407.04620)
Code: [official JAX repo](https://github.com/test-time-training/ttt-lm-jax), [official PyTorch repo](https://github.com/test-time-training/ttt-lm-pytorch)

Sun et al. propose a radical architectural reframing: make the hidden state itself a small model, and let the recurrent update be a self-supervised training step on that model. This turns the hidden state into an explicit learner rather than a passive compressed vector. For your agenda, this is one of the most exciting papers because it directly instantiates the "learn to learn recursively" idea. Instead of treating context as something attention passively reads, TTT layers actively update an internal learner while reading the sequence. That is much closer to human-like accumulation and refinement. The paper also matters computationally: it offers a linear-time alternative to full attention. Conceptually, it says the hidden state need not merely store information; it can be a trainable hypothesis object undergoing continual local learning.

```text
incoming token
      |
      v
hidden state = small model f_t
      |
apply self-supervised update
      |
      v
f_t -> f_{t+1}
      |
      v
use updated model for next prediction
```

### 28. Behrouz et al. (2025) - Titans: Learning to Memorize at Test Time

Primary link: [arXiv](https://arxiv.org/abs/2501.00663)

Titans introduce a neural long-term memory module that complements attention, treating attention as short-term memory and the new memory module as persistent test-time memorization. This is relevant because it explicitly separates accurate local dependency tracking from longer-term compression and storage. That separation is exactly what your rant keeps circling around: few-shot adaptation should not vanish immediately, but neither should every observed example be written blindly into the core weights. Titans are interesting because they add a dedicated place for persistent accumulation that is neither pure prompt context nor full offline training. For your agenda, the paper suggests a promising design principle: use one mechanism for fast local inference over the current context and another mechanism for slower, more persistent memory formation, with explicit gating over what deserves to be remembered.

```text
recent context ----> attention ------\
                                      > combine -> prediction
older history ----> neural memory ---/

attention = short-term memory
memory module = long-term test-time memory
```

## 6. Learned Optimization and Better Inner Learners

### 29. Andrychowicz et al. (2016) - Learning to Learn by Gradient Descent by Gradient Descent

Primary link: [arXiv](https://arxiv.org/abs/1606.04474)

This is the classic learned-optimizer paper. Its direct topic is not transformers or ICL, but it is highly relevant to your intuition that there are many update paths to a correct answer and some are better than others. Andrychowicz et al. show that an optimizer itself can be learned by an outer gradient-based process, typically implemented as an RNN that emits parameter updates. For your agenda, the important conceptual import is that update quality can itself be the object of optimization. You do not have to accept SGD, or whatever inner algorithm emerges in a transformer, as fixed. You can define an outer objective over trajectories and learn an updater that has better inductive biases for the tasks you care about. This is probably the closest classical precursor to "train the model to have good taste about internal updates."

```text
task loss gradients
       |
       v
 learned optimizer RNN
       |
       v
 proposes parameter updates
       |
       v
 outer loop trains optimizer itself
```

### 30. Finn et al. (2017) - Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

Primary link: [arXiv](https://arxiv.org/abs/1703.03400)

MAML is the other essential meta-learning ancestor for your questions. The paper trains model parameters so that a few gradient steps on a new task produce strong generalization. Its central idea is not to encode the whole learner in the forward pass, but to shape initialization so that adaptation trajectories are good. This matters because it provides a clean formalization of "easy to learn from a few examples." In your setting, MAML is relevant less as a literal algorithm for giant LMs and more as a way of thinking: the outer objective should not only care about zero-shot loss, but about whether a small amount of inner evidence produces high-quality generalization. That perspective is a good fit for designing LMs whose in-context updates, test-time patches, or memory writes are optimized for abstraction rather than short-horizon fit.

```text
outer loop:
learn init theta
such that

theta --few task-specific updates--> good generalization

meta-objective rewards updateability
```

## Synthesis Across the 30 Papers

Three broad positions recur across the literature:

- ICL as latent inference: best represented by Xie et al., where the model infers a hidden concept shared by the prompt.
- ICL as implicit optimization: best represented by von Oswald et al., Akyurek et al., Dai et al., and Fu et al., where the forward pass behaves like an optimizer or estimator.
- ICL as dynamic circuit competition: best represented by Olsson et al., Ren et al., Nguyen and Reddy, Park et al., and Wang et al., where multiple sub-circuits or algorithms compete and training dynamics determine which one dominates.

The strongest trend for your research direction is the emerging context-to-weights line:

- Chen et al. make prompt effects convertible into explicit parameter modifications in simplified transformers.
- Dherin et al. argue that transformer blocks already implement temporary low-rank weight writing at inference time.
- Mazzawi et al. and Goldwaser et al. generalize that picture toward deeper and more modern architectures.

The main unresolved gap is still your core obsession:

- Existing papers explain how transient prompt learning can happen.
- Several papers show how prompt effects can be distilled or converted into parameter-like objects.
- But the literature is still weak on how a system should judge whether an internal update is merely fitting the current prompt or actually discovering a higher-level principle worth stabilizing.

That gap suggests the next research questions naturally:

- Can we learn an outer objective over update trajectories, not just final-token correctness?
- Can we score temporary task vectors or fast-weight patches by transfer, compressibility, symmetry, or replay stability?
- Can looped or TTT-style architectures provide explicit inner iterations whose quality can be monitored and improved?
- Can a two-timescale learner use fast prompt-time adaptation to propose hypotheses and slow consolidation to absorb only the hypotheses that survive replay and transfer?

## Code Artifacts Worth Inspecting

Not every paper above ships an official implementation, and several of the theory-heavy papers are paper-only. For the papers that do have code or highly concrete project pages, these are the highest-yield artifacts to inspect first:

- [transformerGD/transformers-learn-in-context-by-gradient-descent](https://github.com/transformerGD/transformers-learn-in-context-by-gradient-descent): the cleanest codebase for the "attention layer as optimizer step" story.
- [OSU-NLP-Group/GrokkedTransformer](https://github.com/OSU-NLP-Group/GrokkedTransformer): useful if you want to see grokking analyzed at the level of circuits and reasoning templates.
- [socialfoundations/tttlm](https://github.com/socialfoundations/tttlm): concrete reference for retrieval plus local test-time adaptation in LMs.
- [test-time-training/ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax): the main implementation of TTT layers in a modern training stack.
- [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch): easier to read for many people than the JAX version if you mainly care about the update rule and hidden-state design.
- [aka.ms/icl](https://aka.ms/icl): Dai et al.'s project page; useful for connecting the meta-optimizer framing to actual experiments and code artifacts.

If I were tracing code in the same order as the papers, I would read the transformerGD notebooks first, then the TTT repos, then Titans-like memory architectures, and only after that return to the newer prompt-to-weights papers.

## Practical Reading Order

If I had to reread this whole area from scratch in a focused order, I would use:

1. Brown et al. 2020
2. Xie et al. 2021
3. Olsson et al. 2022
4. Akyurek et al. 2022
5. von Oswald et al. 2022
6. Dai et al. 2023
7. Hendel et al. 2023
8. Fu et al. 2024
9. Nguyen and Reddy 2024
10. Wang et al. 2024
11. Snell et al. 2022
12. Chen et al. 2024
13. Dherin et al. 2025
14. Mazzawi et al. 2025
15. Goldwaser et al. 2025
16. Gatmiry et al. 2024
17. Sun et al. 2024
18. Behrouz et al. 2025
19. Andrychowicz et al. 2016
20. Finn et al. 2017

That order moves from the phenomenon, to candidate mechanisms, to generalization dynamics, to consolidation, and finally to architectures and meta-objectives for better inner learners.
