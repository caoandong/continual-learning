# Research Notes for ArithCL

These notes explain why the benchmark is shaped the way it is.

## Core benchmark influences

### CLeAR
CLeAR motivates using algorithmic reasoning rather than only vision-style class streams.
The key takeaways adopted here are:
- procedural tasks,
- OOD evaluation,
- task-incremental framing,
- the idea that replay samples may not faithfully represent an abstract concept.

### TRACE
TRACE motivates explicitly measuring damage to earlier abilities during sequential fine-tuning of LLM-like systems. It also reinforces that a benchmark should preserve a held-out general suite rather than only score the newest task.

### TiC-LM
TiC-LM motivates:
- reporting transfer and forgetting over time,
- separating stable knowledge from fast-changing domains,
- benchmarking efficiency against retraining from scratch.

### AttentionSpan
AttentionSpan motivates:
- infinite configurable generators,
- explicit OOD splits by structural difficulty,
- keeping the benchmark clean enough that memorization and rule learning can be disentangled.

## Grokking influences

The arithmetic track is designed as a “fruit-fly” environment for studying whether a system learns reusable rules rather than memorized examples.

The main grokking-inspired design choices are:
- exact correctness,
- infinite procedural data,
- strong length-based OOD tests,
- compositional later tasks,
- adaptation-curve reporting rather than only final accuracy.

## Why the default stage order is addition -> multiplication -> linear combination -> expression

This order intentionally separates:
1. primitive skill acquisition,
2. second primitive acquisition,
3. first recombination test,
4. deeper recombination test.

A learner that truly acquires reusable mechanisms should show:
- low forgetting on addition after multiplication,
- improved pre-training accuracy on linear combination before seeing linear-combination labels,
- reduced examples-to-threshold on linear combination compared with a fresh learner.

## Why include optional non-arithmetic warmup tasks

The user’s research question also asked whether a pre-pretraining style curriculum can encourage later fast adaptation without leaking the downstream arithmetic tasks.

The optional `warmup_transfer_v1` scenario therefore includes:
- value assignment,
- elementary cellular automaton rollout,

before arithmetic stages.

These are not claimed to be the best possible warmups.
They are included because they are:
- procedurally generated,
- clearly non-arithmetic,
- structurally rich,
- cheap to scale.

## What is intentionally not included yet

- task-agnostic class-incremental evaluation without operator/task identifiers
- mechanistic attention-map scoring
- natural-language “general capability” suites
- reinforcement-learning post-training loops
- hidden contradictory knowledge updates

Those would be useful next additions, but the current package focuses on the smallest benchmark that directly operationalizes the desired continual-learning hypothesis.
