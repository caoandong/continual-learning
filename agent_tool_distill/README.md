# Agent Tool-Use Distillation via Low-Rank ΔW

This is a small, readable implementation of the idea discussed in the paper **Learning without training**:

- run a **tool-augmented teacher pass** on context `C'`
- run a **vanilla pass** on context `C`
- compare internal MLP-input activations
- convert the activation gap into a low-rank weight patch
- reuse that patch later on **without tools**

## What is implemented

`agent_tool_distill.py` has three subcommands:

1. `build-corpus`
   - builds a tiny local vector DB from `data/news_snippets.json`
   - default retrieval is TF-IDF so it works offline
   - if you want semantic retrieval, pass a sentence-transformer encoder name

2. `fit-patch`
   - runs a teacher on the `train_new` split
   - runs a vanilla pass on the same questions
   - forces both passes through the same final answer text
   - collects the input activation `a` to the selected MLP linear (`up_proj` by default)
   - computes per-token rank-1 updates
     - `u_i = W Δa_i`
     - `v_i = a_i / ||a_i||^2`
     - `ΔW_i = u_i v_i^T`
   - averages many rank-1 updates and compresses them to rank `r`
   - saves a patch file (`runs/patch.pt`)

3. `evaluate`
   - `vanilla`: no tools, no patch
   - `teacher`: with local tools
   - `patched`: no tools, but with the learned low-rank weight patch applied

## Why this version is approximate

The paper's exact theorem gives a **query-dependent** rank-1 patch for a contextual block.
For future reuse across many questions, we need a **static** patch.
So this repo does the simplest practical approximation:

- compute many exact per-token rank-1 patches on teacher traces
- average them
- compress the result to a small reusable low-rank adapter

That makes it practical for a benchmark, but it is no longer mathematically exact.

## Why `up_proj`?

Nanbeige 4.1-3B is Llama-like, so its MLP has gated projections.
To keep the code short, this repo patches **one clean first-layer analogue**:

- `layer.mlp.up_proj`

If you want a stronger patch, extend the same code to also patch `gate_proj`.

## Benchmark included

The repo includes a tiny reproducible benchmark in `data/`:

- `news_snippets.json` = local post-release AI news snippets
- `news_benchmark.jsonl` = short QA items

Splits:

- `train_new`: new facts to distill
- `eval_new`: held-out rephrasings of those new facts
- `eval_old`: older facts for forgetting checks

## Suggested run

```bash
python agent_tool_distill.py build-corpus \
  --docs data/news_snippets.json \
  --out runs/news_corpus

python agent_tool_distill.py fit-patch \
  --model Nanbeige/Nanbeige4.1-3B \
  --docs data/news_snippets.json \
  --corpus runs/news_corpus \
  --benchmark data/news_benchmark.jsonl \
  --layers -4,-3,-2,-1 \
  --target up_proj \
  --rank 8 \
  --teacher-mode guided \
  --out runs/patch.pt

python agent_tool_distill.py evaluate \
  --model Nanbeige/Nanbeige4.1-3B \
  --docs data/news_snippets.json \
  --corpus runs/news_corpus \
  --benchmark data/news_benchmark.jsonl \
  --mode vanilla \
  --out runs/eval_vanilla.json

python agent_tool_distill.py evaluate \
  --model Nanbeige/Nanbeige4.1-3B \
  --docs data/news_snippets.json \
  --corpus runs/news_corpus \
  --benchmark data/news_benchmark.jsonl \
  --mode teacher \
  --teacher-mode guided \
  --out runs/eval_teacher.json

python agent_tool_distill.py evaluate \
  --model Nanbeige/Nanbeige4.1-3B \
  --docs data/news_snippets.json \
  --corpus runs/news_corpus \
  --benchmark data/news_benchmark.jsonl \
  --mode patched \
  --patch runs/patch.pt \
  --alpha 1.0 \
  --out runs/eval_patched.json
```

## Guided vs auto teacher

`--teacher-mode guided` is the default because it is more reproducible.
It always performs one search over the local corpus, then asks the model to answer.

`--teacher-mode auto` runs a simple function-calling loop and lets the model decide when to call:

- `search_news`
- `read_doc`

Use `auto` if you want more agentic behavior.
Use `guided` if you want stable benchmarking.

## Reading the results

Compare:

- `eval_teacher.json` vs `eval_vanilla.json` on `eval_new`
- `eval_patched.json` vs `eval_vanilla.json` on `eval_new`
- `eval_patched.json` vs `eval_vanilla.json` on `eval_old`

The pattern you want is:

- teacher > vanilla on `eval_new`
- patched > vanilla on `eval_new`
- patched ≈ vanilla on `eval_old`

## Practical knobs

- smaller / safer patch: reduce `--rank` or `--alpha`
- less forgetting: patch fewer late layers, e.g. `--layers -2,-1`
- stronger transfer: patch more layers or extend the code to patch both `up_proj` and `gate_proj`
- faster / lower memory: add `--load-in-4bit`

## Notes

- The local benchmark uses static snippets, not live web fetches, so runs are reproducible.
- The script applies patches at inference time; it does **not** permanently overwrite the base model.
- `build-corpus` was tested here. The full Nanbeige pipeline was not executed in this environment because the model weights and `transformers` runtime were not available locally.
