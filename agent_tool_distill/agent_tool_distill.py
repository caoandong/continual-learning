#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None


TOOL_SYSTEM_PROMPT = (
    "Use tools when needed. Keep search queries short and literal. "
    "Final answers should be short, direct, and citation-free."
)
BASELINE_SYSTEM_PROMPT = "Answer briefly. If unsure, say you are unsure."
DEFAULT_MODEL = "Nanbeige/Nanbeige4.1-3B"
DEFAULT_TARGET = "up_proj"  # cleanest single-matrix analogue of the paper's first MLP weight


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    title: str
    score: float
    text: str
    url: str
    date: str
    source: str


class SimpleVectorIndex:
    """Tiny local vector DB.

    It uses sentence-transformer embeddings when available, and falls back to TF-IDF.
    Corpus size in this benchmark is tiny, so dense matrix search is enough.
    """

    def __init__(self, chunks: List[Dict[str, Any]], encoder_name: str = "tfidf"):
        self.chunks = chunks
        self.encoder_name = encoder_name
        self.encoder_kind = "st" if (encoder_name != "tfidf" and SentenceTransformer is not None) else "tfidf"
        self._model = None
        self._tfidf = None
        self.embeddings = None

    def build(self) -> None:
        texts = [c["text"] for c in self.chunks]
        if self.encoder_kind == "st":
            self._model = SentenceTransformer(self.encoder_name)
            mat = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            self.embeddings = mat.astype(np.float32)
        else:
            if TfidfVectorizer is None:
                raise RuntimeError("Install sentence-transformers or scikit-learn.")
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            mat = self._tfidf.fit_transform(texts)
            self.embeddings = mat

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in self.chunks))
        meta = {"encoder_kind": self.encoder_kind, "encoder_name": self.encoder_name}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.encoder_kind == "st":
            np.save(out_dir / "embeddings.npy", self.embeddings)
        else:
            import pickle

            with open(out_dir / "tfidf.pkl", "wb") as f:
                pickle.dump({"vectorizer": self._tfidf, "matrix": self.embeddings}, f)

    @classmethod
    def load(cls, out_dir: Path) -> "SimpleVectorIndex":
        chunks = [json.loads(line) for line in (out_dir / "chunks.jsonl").read_text().splitlines() if line.strip()]
        meta = json.loads((out_dir / "meta.json").read_text())
        obj = cls(chunks, encoder_name=meta.get("encoder_name", "tfidf"))
        obj.encoder_kind = meta["encoder_kind"]
        if obj.encoder_kind == "st":
            obj.embeddings = np.load(out_dir / "embeddings.npy")
            obj._model = SentenceTransformer(obj.encoder_name)
        else:
            import pickle

            with open(out_dir / "tfidf.pkl", "rb") as f:
                blob = pickle.load(f)
            obj._tfidf = blob["vectorizer"]
            obj.embeddings = blob["matrix"]
        return obj

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.encoder_kind == "st":
            q = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            scores = self.embeddings @ q
            idx = np.argsort(-scores)[:top_k]
            vals = [float(scores[i]) for i in idx]
        else:
            q = self._tfidf.transform([query])
            scores = (self.embeddings @ q.T).toarray().ravel()
            idx = np.argsort(-scores)[:top_k]
            vals = [float(scores[i]) for i in idx]
        out = []
        for i, s in zip(idx, vals):
            c = self.chunks[int(i)]
            out.append(
                SearchResult(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    title=c["title"],
                    score=s,
                    text=c["text"],
                    url=c["url"],
                    date=c["date"],
                    source=c["source"],
                )
            )
        return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9.$\- ]+", "", s)
    return s


def answer_is_correct(pred: str, answers: Sequence[str]) -> bool:
    p = normalize_text(pred)
    for a in answers:
        aa = normalize_text(a)
        if aa and (aa in p or p in aa):
            return True
    return False


def chunk_text(text: str, words_per_chunk: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for i in range(0, len(words), step):
        piece = words[i : i + words_per_chunk]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if i + words_per_chunk >= len(words):
            break
    return chunks


def build_corpus(docs_path: Path, out_dir: Path, encoder_name: str) -> None:
    docs = read_json(docs_path)
    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        for j, text in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}::{j}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "date": doc["date"],
                    "source": doc["source"],
                    "url": doc["url"],
                    "text": text,
                }
            )
    index = SimpleVectorIndex(chunks, encoder_name=encoder_name)
    index.build()
    index.save(out_dir)
    print(f"Built corpus with {len(docs)} docs and {len(chunks)} chunks -> {out_dir}")


# ------------------------- model utilities -------------------------


def load_tokenizer_and_model(model_name: str, load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model_kwargs = dict(trust_remote_code=True, device_map="auto")
    if load_in_4bit:
        model_kwargs.update(dict(load_in_4bit=True))
    else:
        model_kwargs.update(dict(torch_dtype="auto"))
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


def get_layers(model: nn.Module) -> List[nn.Module]:
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "transformer", None),
    ]
    for base in candidates:
        if base is None:
            continue
        if hasattr(base, "layers"):
            return list(base.layers)
        if hasattr(base, "h"):
            return list(base.h)
    raise RuntimeError("Could not locate transformer blocks.")


def get_target_module(layer: nn.Module, target_name: str = DEFAULT_TARGET) -> nn.Module:
    if hasattr(layer, "mlp") and hasattr(layer.mlp, target_name):
        return getattr(layer.mlp, target_name)
    if hasattr(layer, target_name):
        return getattr(layer, target_name)
    raise RuntimeError(f"Could not find target module '{target_name}' in layer {type(layer)}")


def resolve_layer_ids(num_layers: int, spec: str) -> List[int]:
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        ids.append(i if i >= 0 else num_layers + i)
    out = sorted(set(ids))
    for i in out:
        if i < 0 or i >= num_layers:
            raise ValueError(f"Layer id {i} out of range for {num_layers} layers")
    return out


def generate_from_messages(
    tokenizer,
    model,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    gen = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0][input_ids.shape[1] :], skip_special_tokens=True)
    return out.strip()


# ------------------------- tools + agent -------------------------


def tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_news",
                "description": "Search the local AI news vector database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Short search query."},
                        "top_k": {"type": "integer", "description": "Number of results to return."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_doc",
                "description": "Read a full local news document by doc_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "Document id from search results."},
                    },
                    "required": ["doc_id"],
                },
            },
        },
    ]


class LocalTools:
    def __init__(self, corpus_dir: Path, docs_path: Path):
        self.index = SimpleVectorIndex.load(corpus_dir)
        self.docs = {d["id"]: d for d in read_json(docs_path)}

    def search_news(self, query: str, top_k: int = 5) -> str:
        results = self.index.search(query, top_k=top_k)
        payload = []
        for r in results:
            payload.append(
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "date": r.date,
                    "source": r.source,
                    "url": r.url,
                    "score": round(r.score, 4),
                    "snippet": r.text,
                }
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def read_doc(self, doc_id: str) -> str:
        doc = self.docs[doc_id]
        payload = {
            "doc_id": doc["id"],
            "title": doc["title"],
            "date": doc["date"],
            "source": doc["source"],
            "url": doc["url"],
            "text": doc["text"],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


_JSON_BLOCK = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for match in _JSON_BLOCK.finditer(text):
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            if not isinstance(obj["arguments"], dict):
                continue
            calls.append(obj)
    return calls


def run_tool_agent(
    tokenizer,
    model,
    tools_backend: LocalTools,
    question: str,
    max_steps: int = 4,
    temperature: float = 0.0,
    fallback_search: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = tool_specs()

    for step in range(max_steps):
        reply = generate_from_messages(tokenizer, model, messages, tools=tools, temperature=temperature)
        calls = extract_tool_calls(reply)
        messages.append({"role": "assistant", "content": reply})
        if not calls:
            if step == 0 and fallback_search:
                # robust fallback for reproducible runs
                fallback = tools_backend.search_news(question, top_k=4)
                messages.append({"role": "tool", "content": fallback})
                continue
            return reply, messages

        for call in calls:
            name = call["name"]
            args = call.get("arguments", {})
            if name == "search_news":
                result = tools_backend.search_news(args.get("query", question), int(args.get("top_k", 4)))
            elif name == "read_doc":
                result = tools_backend.read_doc(args["doc_id"])
            else:
                result = json.dumps({"error": f"unknown tool: {name}"})
            messages.append({"role": "tool", "content": result})

    final = generate_from_messages(tokenizer, model, messages, tools=tools, temperature=temperature)
    messages.append({"role": "assistant", "content": final})
    return final, messages


def run_guided_teacher(
    tokenizer,
    model,
    tools_backend: LocalTools,
    question: str,
    temperature: float = 0.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Deterministic teacher: always search once, then ask the model to answer.

    This is less agentic than run_tool_agent, but much more reproducible. It is the default
    for distillation. The retrieved snippets still act as the tool-use trace context C'.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    search_blob = tools_backend.search_news(question, top_k=4)
    messages.append({"role": "assistant", "content": '{"name": "search_news", "arguments": {"query": ' + json.dumps(question) + '}}'})
    messages.append({"role": "tool", "content": search_blob})
    answer = generate_from_messages(tokenizer, model, messages, tools=tool_specs(), temperature=temperature)
    messages.append({"role": "assistant", "content": answer})
    return answer, messages


def run_vanilla(tokenizer, model, question: str, temperature: float = 0.0) -> Tuple[str, List[Dict[str, Any]]]:
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = generate_from_messages(tokenizer, model, messages, tools=None, temperature=temperature)
    messages.append({"role": "assistant", "content": answer})
    return answer, messages


# ------------------------- tracing + patch fitting -------------------------


def build_teacher_forcing_example(
    tokenizer,
    prefix_messages: List[Dict[str, Any]],
    answer_text: str,
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[torch.Tensor, List[int]]:
    prefix = tokenizer.apply_chat_template(prefix_messages, tools=tools, add_generation_prompt=True, tokenize=False)
    full = prefix + answer_text + tokenizer.eos_token
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids[0]
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if len(answer_ids) == 0:
        raise ValueError("Answer has zero tokens after tokenization.")
    # positions whose hidden states predict each answer token
    pred_positions = list(range(len(prefix_ids) - 1, len(prefix_ids) - 1 + len(answer_ids)))
    return full_ids.unsqueeze(0), pred_positions


@torch.no_grad()
def capture_linear_inputs(model: nn.Module, input_ids: torch.Tensor, layer_ids: List[int], target_name: str) -> Dict[int, torch.Tensor]:
    layers = get_layers(model)
    storage: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer_id in layer_ids:
        module = get_target_module(layers[layer_id], target_name)

        def make_hook(i: int):
            def hook(_module, inputs):
                storage[i] = inputs[0].detach().cpu()[0]  # [seq, hidden]
            return hook

        hooks.append(module.register_forward_pre_hook(make_hook(layer_id)))

    _ = model(input_ids=input_ids.to(model.device), use_cache=False)
    for h in hooks:
        h.remove()
    return storage


@torch.no_grad()
def linear_map_no_bias(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        w = module.weight.detach()
        return F.linear(x.to(w.device, dtype=w.dtype), w, None).float().cpu()
    # quantized fallback: assume no bias, which is true for Llama/Qwen up_proj
    y = module(x.to(next(module.parameters()).device).unsqueeze(0)).squeeze(0)
    return y.float().cpu()


class LowRankPatchedLinear(nn.Module):
    def __init__(self, base: nn.Module, left: torch.Tensor, right: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        dx = (x.to(self.right.dtype) @ self.right) @ self.left.T
        return y + self.alpha * dx.to(y.dtype)


def compress_outer_sums(u_list: List[torch.Tensor], v_list: List[torch.Tensor], rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not u_list:
        raise ValueError("No factors collected.")
    B = torch.stack(u_list, dim=1) / len(u_list)  # [out, m]
    A = torch.stack(v_list, dim=1)  # [in, m]
    qB, rB = torch.linalg.qr(B, mode="reduced")
    qA, rA = torch.linalg.qr(A, mode="reduced")
    core = rB @ rA.T
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    r = min(rank, int((S > 1e-8).sum().item()), S.numel())
    if r == 0:
        r = 1
    sqrtS = torch.sqrt(S[:r])
    left = qB @ (U[:, :r] * sqrtS.unsqueeze(0))
    right = qA @ (Vh[:r, :].T * sqrtS.unsqueeze(0))
    return left.contiguous(), right.contiguous()


def fit_patch(
    tokenizer,
    model,
    docs_path: Path,
    corpus_dir: Path,
    benchmark_path: Path,
    out_path: Path,
    layer_spec: str,
    target_name: str,
    rank: int,
    teacher_mode: str,
    temperature: float,
) -> None:
    layers = get_layers(model)
    layer_ids = resolve_layer_ids(len(layers), layer_spec)
    tools_backend = LocalTools(corpus_dir=corpus_dir, docs_path=docs_path)
    benchmark = read_jsonl(benchmark_path)
    train_items = [x for x in benchmark if x["split"] == "train_new"]

    u_bank: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_ids}
    v_bank: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_ids}
    trace_rows = []

    for item in train_items:
        question = item["question"]
        if teacher_mode == "guided":
            teacher_answer, teacher_messages = run_guided_teacher(tokenizer, model, tools_backend, question, temperature=temperature)
        else:
            teacher_answer, teacher_messages = run_tool_agent(tokenizer, model, tools_backend, question, temperature=temperature)
        _, vanilla_messages = run_vanilla(tokenizer, model, question, temperature=temperature)

        teacher_prefix = teacher_messages[:-1]
        vanilla_prefix = vanilla_messages[:-1]
        teacher_ids, teacher_pos = build_teacher_forcing_example(tokenizer, teacher_prefix, teacher_answer, tools=tool_specs())
        vanilla_ids, vanilla_pos = build_teacher_forcing_example(tokenizer, vanilla_prefix, teacher_answer, tools=None)

        teacher_acts = capture_linear_inputs(model, teacher_ids, layer_ids, target_name=target_name)
        vanilla_acts = capture_linear_inputs(model, vanilla_ids, layer_ids, target_name=target_name)

        for layer_id in layer_ids:
            module = get_target_module(get_layers(model)[layer_id], target_name)
            for p_t, p_v in zip(teacher_pos, vanilla_pos):
                a_t = teacher_acts[layer_id][p_t]
                a_v = vanilla_acts[layer_id][p_v]
                delta_a = a_t - a_v
                denom = float(torch.dot(a_v, a_v).item()) + 1e-8
                u = linear_map_no_bias(module, delta_a)
                v = a_v.float().cpu() / denom
                u_bank[layer_id].append(u)
                v_bank[layer_id].append(v)

        trace_rows.append(
            {
                "id": item["id"],
                "question": question,
                "teacher_answer": teacher_answer,
                "gold": item["answers"],
                "teacher_correct": answer_is_correct(teacher_answer, item["answers"]),
            }
        )
        print(f"traced {item['id']}: teacher_correct={trace_rows[-1]['teacher_correct']} answer={teacher_answer!r}")

    patch = {
        "model_name": getattr(model.config, "_name_or_path", DEFAULT_MODEL),
        "target_name": target_name,
        "layer_ids": layer_ids,
        "rank": rank,
        "teacher_mode": teacher_mode,
        "patches": {},
        "trace_rows": trace_rows,
    }
    for layer_id in layer_ids:
        left, right = compress_outer_sums(u_bank[layer_id], v_bank[layer_id], rank=rank)
        patch["patches"][str(layer_id)] = {"left": left, "right": right}
        print(f"layer {layer_id}: collected {len(u_bank[layer_id])} rank-1 factors -> rank-{rank} patch")

    torch.save(patch, out_path)
    print(f"saved patch -> {out_path}")


def apply_patch(model: nn.Module, patch_path: Path, alpha: float = 1.0) -> List[LowRankPatchedLinear]:
    patch = torch.load(patch_path, map_location="cpu")
    target_name = patch["target_name"]
    layers = get_layers(model)
    wrappers: List[LowRankPatchedLinear] = []
    for k, blob in patch["patches"].items():
        layer_id = int(k)
        layer = layers[layer_id]
        if not hasattr(layer, "mlp"):
            raise RuntimeError("This simple patcher expects layer.mlp.<target>")
        base = getattr(layer.mlp, target_name)
        device = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        wrapper = LowRankPatchedLinear(
            base=base,
            left=blob["left"].to(device=device, dtype=dtype),
            right=blob["right"].to(device=device, dtype=dtype),
            alpha=alpha,
        )
        setattr(layer.mlp, target_name, wrapper)
        wrappers.append(wrapper)
    return wrappers


# ------------------------- evaluation -------------------------


def evaluate(
    tokenizer,
    model,
    docs_path: Path,
    corpus_dir: Path,
    benchmark_path: Path,
    out_path: Path,
    mode: str,
    temperature: float,
    teacher_mode: str,
) -> None:
    tools_backend = LocalTools(corpus_dir=corpus_dir, docs_path=docs_path)
    benchmark = read_jsonl(benchmark_path)
    rows = []
    for item in benchmark:
        split = item["split"]
        if mode == "teacher":
            if teacher_mode == "guided":
                pred, _ = run_guided_teacher(tokenizer, model, tools_backend, item["question"], temperature=temperature)
            else:
                pred, _ = run_tool_agent(tokenizer, model, tools_backend, item["question"], temperature=temperature)
        else:
            pred, _ = run_vanilla(tokenizer, model, item["question"], temperature=temperature)
        correct = answer_is_correct(pred, item["answers"])
        rows.append({"id": item["id"], "split": split, "question": item["question"], "pred": pred, "gold": item["answers"], "correct": correct})
        print(f"{mode:>7s} | {item['id']} | correct={correct} | pred={pred!r}")

    by_split = {}
    for split in sorted(set(r["split"] for r in rows)):
        subset = [r for r in rows if r["split"] == split]
        by_split[split] = sum(r["correct"] for r in subset) / max(1, len(subset))
    report = {"mode": mode, "scores": by_split, "rows": rows}
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["scores"], indent=2))
    print(f"saved eval -> {out_path}")


# ------------------------- CLI -------------------------


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distill tool-use traces into low-rank weight patches.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-corpus")
    p_build.add_argument("--docs", type=Path, default=Path("data/news_snippets.json"))
    p_build.add_argument("--out", type=Path, default=Path("runs/news_corpus"))
    p_build.add_argument("--encoder", default="tfidf")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--docs", type=Path, default=Path("data/news_snippets.json"))
    common.add_argument("--corpus", type=Path, default=Path("runs/news_corpus"))
    common.add_argument("--benchmark", type=Path, default=Path("data/news_benchmark.jsonl"))
    common.add_argument("--temperature", type=float, default=0.0)
    common.add_argument("--load-in-4bit", action="store_true")

    p_fit = sub.add_parser("fit-patch", parents=[common])
    p_fit.add_argument("--layers", default="-4,-3,-2,-1")
    p_fit.add_argument("--target", default=DEFAULT_TARGET)
    p_fit.add_argument("--rank", type=int, default=8)
    p_fit.add_argument("--teacher-mode", choices=["guided", "auto"], default="guided")
    p_fit.add_argument("--out", type=Path, default=Path("runs/patch.pt"))

    p_eval = sub.add_parser("evaluate", parents=[common])
    p_eval.add_argument("--mode", choices=["vanilla", "teacher", "patched"], default="vanilla")
    p_eval.add_argument("--teacher-mode", choices=["guided", "auto"], default="guided")
    p_eval.add_argument("--patch", type=Path, default=Path("runs/patch.pt"))
    p_eval.add_argument("--alpha", type=float, default=1.0)
    p_eval.add_argument("--out", type=Path, default=Path("runs/eval.json"))
    return p


def main() -> None:
    args = make_parser().parse_args()
    if args.cmd == "build-corpus":
        build_corpus(args.docs, args.out, args.encoder)
        return

    tokenizer, model = load_tokenizer_and_model(args.model, load_in_4bit=args.load_in_4bit)

    if args.cmd == "fit-patch":
        fit_patch(
            tokenizer=tokenizer,
            model=model,
            docs_path=args.docs,
            corpus_dir=args.corpus,
            benchmark_path=args.benchmark,
            out_path=args.out,
            layer_spec=args.layers,
            target_name=args.target,
            rank=args.rank,
            teacher_mode=args.teacher_mode,
            temperature=args.temperature,
        )
        return

    if args.cmd == "evaluate":
        if args.mode == "patched":
            apply_patch(model, args.patch, alpha=args.alpha)
            mode = "patched"
        else:
            mode = args.mode
        eval_mode = "vanilla" if mode == "patched" else mode
        evaluate(
            tokenizer=tokenizer,
            model=model,
            docs_path=args.docs,
            corpus_dir=args.corpus,
            benchmark_path=args.benchmark,
            out_path=args.out,
            mode=eval_mode,
            temperature=args.temperature,
            teacher_mode=args.teacher_mode,
        )
        return


if __name__ == "__main__":
    main()
