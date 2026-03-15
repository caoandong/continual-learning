#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None


TOOL_SYSTEM_PROMPT = (
    "Use tools when needed. Keep search queries short and literal. "
    "Final answers should be short, direct, and citation-free."
)
BASELINE_SYSTEM_PROMPT = "Answer briefly. If unsure, say you are unsure."
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_SWEEP_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]
DEFAULT_TARGET = "up_proj"
DEFAULT_REASONING_MODE = "non_thinking"
QWEN3_REASONING_MODES = ("non_thinking", "thinking")
SUPPORTED_DEVICES = ("auto", "mps", "cpu", "cuda")
REPO_ROOT = Path(__file__).resolve().parent
PREFERRED_TRANSFORMERS_SRC = Path("/Volumes/SB-XTM5/flair/software/transformers")
DEFAULT_TRANSFORMERS_SRC = PREFERRED_TRANSFORMERS_SRC if PREFERRED_TRANSFORMERS_SRC.exists() else None


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_DOCS_PATH = first_existing_path(
    REPO_ROOT / "news_snippets.json",
    REPO_ROOT / "agent_tool_distill" / "data" / "news_snippets.json",
)
DEFAULT_BENCHMARK_PATH = first_existing_path(
    REPO_ROOT / "news_benchmark.jsonl",
    REPO_ROOT / "agent_tool_distill" / "data" / "news_benchmark.jsonl",
)
DEFAULT_CORPUS_DIR = REPO_ROOT / "runs" / "news_corpus"
DEFAULT_PATCH_PATH = REPO_ROOT / "runs" / "patch.pt"
DEFAULT_EVAL_PATH = REPO_ROOT / "runs" / "eval.json"
DEFAULT_SWEEP_ROOT = REPO_ROOT / "runs" / "qwen3"

QWEN3_RESPONSE_SCHEMA = {
    "x-regex": r"^(?:(?:<think>)?\s*(?P<thinking>.+?)\s*</think>)?\s*(?:<tool_call>(?P<tool_calls>.*?)\s*</tool_call>)?\s*(?P<content>.+?)?\s*$",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "thinking": {"type": "string"},
        "tool_calls": {
            "x-regex-iterator": r"^(.*)$",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"<function=(\w+)>"},
                            "arguments": {
                                "type": "object",
                                "x-regex-key-value": r"<parameter=(?P<key>\w+)>\n(?P<value>.*?)\n</parameter>",
                                "additionalProperties": {
                                    "x-parser": "json",
                                    "x-parser-args": {"allow_non_json": True},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

_JSON_BLOCK = re.compile(r"\{[\s\S]*?\}", re.MULTILINE)
_QWEN_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_QWEN_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN_FUNCTION_BLOCK = re.compile(r"<function=(?P<name>\w+)>\s*(?P<body>.*?)\s*</function>", re.DOTALL)
_QWEN_PARAMETER_BLOCK = re.compile(r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)


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


@dataclass
class AssistantResponse:
    raw_text: str
    content: str
    thinking: Optional[str]
    tool_calls: List[Dict[str, Any]]


@dataclass
class ConversationRun:
    answer: str
    messages: List[Dict[str, Any]]
    response: AssistantResponse


@dataclass
class RuntimeConfig:
    reasoning_mode: str = DEFAULT_REASONING_MODE
    temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None
    device: str = "auto"
    transformers_src: Optional[Path] = DEFAULT_TRANSFORMERS_SRC


@dataclass
class TransformersRuntime:
    version: str
    import_path: str


class SimpleVectorIndex:
    """Tiny local vector DB backed by sentence-transformers or TF-IDF."""

    def __init__(self, chunks: List[Dict[str, Any]], encoder_name: str = "tfidf"):
        self.chunks = chunks
        self.encoder_name = encoder_name
        self.encoder_kind = "st" if encoder_name != "tfidf" else "tfidf"
        self._model = None
        self._tfidf = None
        self.embeddings = None

    def build(self) -> None:
        texts = [c["text"] for c in self.chunks]
        if self.encoder_kind == "st":
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers is required for dense retrieval.") from exc
            self._model = SentenceTransformer(self.encoder_name)
            mat = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            self.embeddings = mat.astype(np.float32)
        else:
            if TfidfVectorizer is None:
                raise RuntimeError("Install sentence-transformers or scikit-learn.")
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            self.embeddings = self._tfidf.fit_transform(texts)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in self.chunks))
        meta = {"encoder_kind": self.encoder_kind, "encoder_name": self.encoder_name}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if self.encoder_kind == "st":
            np.save(out_dir / "embeddings.npy", self.embeddings)
        else:
            import pickle

            with open(out_dir / "tfidf.pkl", "wb") as handle:
                pickle.dump({"vectorizer": self._tfidf, "matrix": self.embeddings}, handle)

    @classmethod
    def load(cls, out_dir: Path) -> "SimpleVectorIndex":
        chunks = [json.loads(line) for line in (out_dir / "chunks.jsonl").read_text().splitlines() if line.strip()]
        meta = json.loads((out_dir / "meta.json").read_text())
        obj = cls(chunks, encoder_name=meta.get("encoder_name", "tfidf"))
        obj.encoder_kind = meta["encoder_kind"]
        if obj.encoder_kind == "st":
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("sentence-transformers is required for dense retrieval.") from exc
            obj.embeddings = np.load(out_dir / "embeddings.npy")
            obj._model = SentenceTransformer(obj.encoder_name)
        else:
            import pickle

            with open(out_dir / "tfidf.pkl", "rb") as handle:
                blob = pickle.load(handle)
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
        return [
            SearchResult(
                chunk_id=self.chunks[int(i)]["chunk_id"],
                doc_id=self.chunks[int(i)]["doc_id"],
                title=self.chunks[int(i)]["title"],
                score=s,
                text=self.chunks[int(i)]["text"],
                url=self.chunks[int(i)]["url"],
                date=self.chunks[int(i)]["date"],
                source=self.chunks[int(i)]["source"],
            )
            for i, s in zip(idx, vals)
        ]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9.$\- ]+", "", s)
    return s


def answer_is_correct(pred: str, answers: Sequence[str]) -> bool:
    p = normalize_text(pred)
    for answer in answers:
        normalized = normalize_text(answer)
        if normalized and (normalized in p or p in normalized):
            return True
    return False


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=None)
def read_local_model_metadata(model_name: str) -> Dict[str, Any]:
    path = Path(model_name).expanduser()
    if not path.exists():
        return {}

    metadata: Dict[str, Any] = {}
    config_path = path / "config.json"
    if config_path.exists():
        try:
            metadata.update(json.loads(config_path.read_text()))
        except Exception:
            pass

    readme_path = path / "README.md"
    if readme_path.exists():
        try:
            metadata["readme_text"] = readme_path.read_text(errors="ignore")[:8192]
        except Exception:
            pass
    return metadata


def is_qwen3_model(model_name: str) -> bool:
    lowered = model_name.lower()
    if "qwen3" in lowered:
        return True

    metadata = read_local_model_metadata(model_name)
    if metadata.get("model_type") == "qwen3":
        return True
    if any("qwen3" in str(arch).lower() for arch in metadata.get("architectures", [])):
        return True
    return "qwen3" in str(metadata.get("readme_text", "")).lower()


def is_qwen3_instruct_2507_model(model_name: str) -> bool:
    lowered = model_name.lower()
    if "qwen3" in lowered and "instruct-2507" in lowered:
        return True
    return "instruct-2507" in str(read_local_model_metadata(model_name).get("readme_text", "")).lower()


def chunk_text(text: str, words_per_chunk: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for index in range(0, len(words), step):
        piece = words[index : index + words_per_chunk]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if index + words_per_chunk >= len(words):
            break
    return chunks


def build_corpus(docs_path: Path, out_dir: Path, encoder_name: str) -> None:
    docs = read_json(docs_path)
    chunks: List[Dict[str, Any]] = []
    for doc in docs:
        for index, text in enumerate(chunk_text(doc["text"])):
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}::{index}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "date": doc["date"],
                    "source": doc["source"],
                    "url": doc["url"],
                    "text": text,
                }
            )
    vector_index = SimpleVectorIndex(chunks, encoder_name=encoder_name)
    vector_index.build()
    vector_index.save(out_dir)
    print(f"Built corpus with {len(docs)} docs and {len(chunks)} chunks -> {out_dir}")


def ensure_corpus(corpus_dir: Path, docs_path: Path, encoder_name: str) -> None:
    required = [corpus_dir / "meta.json", corpus_dir / "chunks.jsonl"]
    if all(path.exists() for path in required):
        return
    build_corpus(docs_path=docs_path, out_dir=corpus_dir, encoder_name=encoder_name)


def resolve_transformers_src(transformers_src: Optional[Path]) -> Optional[Path]:
    if transformers_src is None:
        return None
    root = transformers_src.resolve()
    if (root / "src" / "transformers").exists():
        return root / "src"
    if (root / "transformers").exists():
        return root
    raise FileNotFoundError(f"Could not find a transformers package under {transformers_src}")


def load_transformers_module(transformers_src: Optional[Path]) -> Tuple[Any, Optional[Path]]:
    resolved_src = resolve_transformers_src(transformers_src)
    existing = sys.modules.get("transformers")
    if existing is not None:
        if resolved_src is not None:
            module_path = Path(existing.__file__).resolve()
            if resolved_src not in module_path.parents and module_path.parent != resolved_src:
                raise RuntimeError(
                    "transformers is already imported from a different location. "
                    f"requested={resolved_src} active={module_path}"
                )
        return existing, resolved_src

    if resolved_src is not None and str(resolved_src) not in sys.path:
        sys.path.insert(0, str(resolved_src))
        importlib.invalidate_caches()

    transformers = importlib.import_module("transformers")
    from packaging.version import Version

    if Version(transformers.__version__) < Version("4.57.0"):
        raise RuntimeError(
            f"transformers>={Version('4.57.0')} is required for Qwen3 support; found {transformers.__version__}"
        )
    return transformers, resolved_src


def validate_requested_device(device: str) -> str:
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device '{device}'. Expected one of {SUPPORTED_DEVICES}.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return device


def install_qwen3_response_schema(tokenizer: Any, model_name: str) -> None:
    if is_qwen3_model(model_name) and getattr(tokenizer, "response_schema", None) is None:
        tokenizer.response_schema = QWEN3_RESPONSE_SCHEMA


def get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def load_tokenizer_and_model(
    model_name: str,
    runtime: RuntimeConfig,
    load_in_4bit: bool = False,
) -> Tuple[Any, nn.Module, TransformersRuntime]:
    validate_requested_device(runtime.device)
    if load_in_4bit and runtime.device in {"cpu", "mps"}:
        raise ValueError("--load-in-4bit is only supported on Linux CUDA environments.")

    transformers, _ = load_transformers_module(runtime.transformers_src)
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_qwen3_model(model_name):
        tokenizer.padding_side = "left"
    install_qwen3_response_schema(tokenizer, model_name)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["dtype"] = "auto"

    if runtime.device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if runtime.device != "auto":
        model.to(torch.device(runtime.device))
    model.eval()

    runtime_info = TransformersRuntime(
        version=transformers.__version__,
        import_path=str(Path(transformers.__file__).resolve()),
    )
    return tokenizer, model, runtime_info


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
        value = int(part)
        ids.append(value if value >= 0 else num_layers + value)
    out = sorted(set(ids))
    for layer_id in out:
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(f"Layer id {layer_id} out of range for {num_layers} layers")
    return out


def make_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def normalize_tool_calls(tool_calls: Sequence[Any]) -> List[Dict[str, Any]]:
    normalized = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if "function" in call and isinstance(call["function"], dict):
            name = call["function"].get("name")
            arguments = call["function"].get("arguments", {})
        else:
            name = call.get("name")
            arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        if not name or not isinstance(arguments, dict):
            continue
        normalized.append(make_tool_call(name, arguments))
    return normalized


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for match in _JSON_BLOCK.finditer(text):
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj and isinstance(obj["arguments"], dict):
            calls.append(make_tool_call(obj["name"], obj["arguments"]))
    return calls


def extract_qwen_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for block_match in _QWEN_TOOL_CALL_BLOCK.finditer(text):
        block = block_match.group(1)
        for fn_match in _QWEN_FUNCTION_BLOCK.finditer(block):
            arguments = {}
            for param_match in _QWEN_PARAMETER_BLOCK.finditer(fn_match.group("body")):
                value = param_match.group("value").strip()
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                arguments[param_match.group("key")] = parsed_value
            calls.append(make_tool_call(fn_match.group("name"), arguments))
    return calls


def parse_assistant_response(tokenizer: Any, raw_text: str, model_name: str) -> AssistantResponse:
    raw_text = raw_text.strip()
    install_qwen3_response_schema(tokenizer, model_name)

    schema = getattr(tokenizer, "response_schema", None)
    parsed = None
    if hasattr(tokenizer, "parse_response"):
        try:
            parsed = tokenizer.parse_response(raw_text, schema=schema) if schema is not None else tokenizer.parse_response(raw_text)
        except TypeError:
            parsed = tokenizer.parse_response(raw_text)
        except Exception:
            parsed = None

    if isinstance(parsed, dict):
        tool_calls = normalize_tool_calls(parsed.get("tool_calls") or [])
        content = parsed.get("content")
        if not isinstance(content, str):
            content = "" if tool_calls else raw_text
        thinking = parsed.get("thinking")
        if isinstance(thinking, str):
            thinking = thinking.strip() or None
        else:
            thinking = None
        return AssistantResponse(
            raw_text=raw_text,
            content=content.strip(),
            thinking=thinking,
            tool_calls=tool_calls,
        )

    thinking = None
    text_without_thinking = raw_text
    think_match = _QWEN_THINK_BLOCK.search(raw_text)
    if think_match:
        thinking = think_match.group(1).strip() or None
        text_without_thinking = _QWEN_THINK_BLOCK.sub("", raw_text).strip()

    tool_calls = extract_qwen_tool_calls(text_without_thinking)
    if not tool_calls:
        tool_calls = extract_tool_calls(text_without_thinking)

    content = text_without_thinking
    if tool_calls:
        if "<tool_call>" in text_without_thinking:
            content = _QWEN_TOOL_CALL_BLOCK.sub("", text_without_thinking).strip()
        else:
            stripped = text_without_thinking.strip()
            content = "" if stripped.startswith("{") and stripped.endswith("}") else stripped

    return AssistantResponse(
        raw_text=raw_text,
        content=content.strip(),
        thinking=thinking,
        tool_calls=tool_calls,
    )


def answer_text(response: AssistantResponse) -> str:
    return response.content or response.raw_text


def assistant_message_for_history(response: AssistantResponse) -> Dict[str, Any]:
    content = response.content if response.tool_calls else answer_text(response)
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    return message


def resolve_generation_kwargs(model_name: str, runtime: RuntimeConfig) -> Dict[str, Any]:
    if is_qwen3_instruct_2507_model(model_name):
        if runtime.reasoning_mode != "non_thinking":
            raise ValueError(
                f"{model_name} is a non-thinking Qwen3 Instruct checkpoint; "
                "use --reasoning-mode non_thinking."
            )
        kwargs = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_new_tokens": runtime.max_new_tokens or 512,
        }
        if runtime.temperature is not None:
            kwargs["temperature"] = runtime.temperature
        return kwargs

    if is_qwen3_model(model_name):
        presets = {
            "non_thinking": {
                "enable_thinking": False,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            },
            "thinking": {
                "enable_thinking": True,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            },
        }
        kwargs = dict(presets[runtime.reasoning_mode])
        if runtime.temperature is not None:
            kwargs["temperature"] = runtime.temperature
        kwargs["max_new_tokens"] = runtime.max_new_tokens or 512
        return kwargs

    temperature = 0.0 if runtime.temperature is None else runtime.temperature
    kwargs = {
        "max_new_tokens": runtime.max_new_tokens or 160,
        "do_sample": temperature > 0,
        "top_p": 1.0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    return kwargs


def generate_from_messages(
    tokenizer: Any,
    model: nn.Module,
    model_name: str,
    messages: List[Dict[str, Any]],
    runtime: RuntimeConfig,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AssistantResponse:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    device = get_model_device(model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_kwargs = resolve_generation_kwargs(model_name, runtime)
    generation_kwargs.update(
        {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
    )

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    raw_text = tokenizer.decode(generated[0][input_ids.shape[1] :], skip_special_tokens=True).strip()
    return parse_assistant_response(tokenizer=tokenizer, raw_text=raw_text, model_name=model_name)


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
        self.docs = {doc["id"]: doc for doc in read_json(docs_path)}

    def search_news(self, query: str, top_k: int = 5) -> str:
        results = self.index.search(query, top_k=top_k)
        payload = [
            {
                "doc_id": result.doc_id,
                "title": result.title,
                "date": result.date,
                "source": result.source,
                "url": result.url,
                "score": round(result.score, 4),
                "snippet": result.text,
            }
            for result in results
        ]
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


def execute_tool_call(call: Dict[str, Any], tools_backend: LocalTools, question: str) -> Tuple[str, str]:
    fn = call.get("function", {})
    name = fn.get("name")
    arguments = fn.get("arguments", {})
    if name == "search_news":
        query = arguments.get("query", question)
        top_k = int(arguments.get("top_k", 4))
        return name, tools_backend.search_news(query=query, top_k=top_k)
    if name == "read_doc":
        return name, tools_backend.read_doc(arguments["doc_id"])
    return name or "unknown", json.dumps({"error": f"unknown tool: {name}"})


def make_tool_message(name: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "name": name, "content": content}


def run_tool_agent(
    tokenizer: Any,
    model: nn.Module,
    model_name: str,
    tools_backend: LocalTools,
    question: str,
    runtime: RuntimeConfig,
    max_steps: int = 4,
    fallback_search: bool = True,
) -> ConversationRun:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = tool_specs()

    for step in range(max_steps):
        response = generate_from_messages(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            messages=messages,
            runtime=runtime,
            tools=tools,
        )
        if response.tool_calls:
            messages.append(assistant_message_for_history(response))
            for call in response.tool_calls:
                tool_name, tool_result = execute_tool_call(call, tools_backend, question)
                messages.append(make_tool_message(tool_name, tool_result))
            continue

        if step == 0 and fallback_search:
            fallback_call = make_tool_call("search_news", {"query": question, "top_k": 4})
            messages.append({"role": "assistant", "content": "", "tool_calls": [fallback_call]})
            messages.append(make_tool_message("search_news", tools_backend.search_news(question, top_k=4)))
            continue

        messages.append(assistant_message_for_history(response))
        return ConversationRun(answer=answer_text(response), messages=messages, response=response)

    response = generate_from_messages(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        messages=messages,
        runtime=runtime,
        tools=tools,
    )
    messages.append(assistant_message_for_history(response))
    return ConversationRun(answer=answer_text(response), messages=messages, response=response)


def run_guided_teacher(
    tokenizer: Any,
    model: nn.Module,
    model_name: str,
    tools_backend: LocalTools,
    question: str,
    runtime: RuntimeConfig,
) -> ConversationRun:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    search_call = make_tool_call("search_news", {"query": question, "top_k": 4})
    search_blob = tools_backend.search_news(question, top_k=4)
    messages.append({"role": "assistant", "content": "", "tool_calls": [search_call]})
    messages.append(make_tool_message("search_news", search_blob))
    response = generate_from_messages(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        messages=messages,
        runtime=runtime,
        tools=tool_specs(),
    )
    messages.append(assistant_message_for_history(response))
    return ConversationRun(answer=answer_text(response), messages=messages, response=response)


def run_vanilla(
    tokenizer: Any,
    model: nn.Module,
    model_name: str,
    question: str,
    runtime: RuntimeConfig,
) -> ConversationRun:
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = generate_from_messages(
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        messages=messages,
        runtime=runtime,
        tools=None,
    )
    messages.append(assistant_message_for_history(response))
    return ConversationRun(answer=answer_text(response), messages=messages, response=response)


def build_teacher_forcing_example(
    tokenizer: Any,
    prefix_messages: List[Dict[str, Any]],
    answer_text_value: str,
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[torch.Tensor, List[int]]:
    prefix = tokenizer.apply_chat_template(prefix_messages, tools=tools, add_generation_prompt=True, tokenize=False)
    full = prefix + answer_text_value + tokenizer.eos_token
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids[0]
    answer_ids = tokenizer(answer_text_value, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if len(answer_ids) == 0:
        raise ValueError("Answer has zero tokens after tokenization.")
    pred_positions = list(range(len(prefix_ids) - 1, len(prefix_ids) - 1 + len(answer_ids)))
    return full_ids.unsqueeze(0), pred_positions


@torch.no_grad()
def capture_linear_inputs(model: nn.Module, input_ids: torch.Tensor, layer_ids: List[int], target_name: str) -> Dict[int, torch.Tensor]:
    layers = get_layers(model)
    storage: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer_id in layer_ids:
        module = get_target_module(layers[layer_id], target_name)

        def make_hook(index: int):
            def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
                storage[index] = inputs[0].detach().cpu()[0]

            return hook

        hooks.append(module.register_forward_pre_hook(make_hook(layer_id)))

    _ = model(input_ids=input_ids.to(get_model_device(model)), use_cache=False)
    for hook in hooks:
        hook.remove()
    return storage


@torch.no_grad()
def linear_map_no_bias(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        weight = module.weight.detach()
        return F.linear(x.to(weight.device, dtype=weight.dtype), weight, None).float().cpu()
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
    b_mat = torch.stack(u_list, dim=1) / len(u_list)
    a_mat = torch.stack(v_list, dim=1)
    q_b, r_b = torch.linalg.qr(b_mat, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_mat, mode="reduced")
    core = r_b @ r_a.T
    u, singular_values, vh = torch.linalg.svd(core, full_matrices=False)
    rank_limit = min(rank, int((singular_values > 1e-8).sum().item()), singular_values.numel())
    if rank_limit == 0:
        rank_limit = 1
    sqrt_s = torch.sqrt(singular_values[:rank_limit])
    left = q_b @ (u[:, :rank_limit] * sqrt_s.unsqueeze(0))
    right = q_a @ (vh[:rank_limit, :].T * sqrt_s.unsqueeze(0))
    return left.contiguous(), right.contiguous()


def build_artifact_metadata(
    *,
    transformers_runtime: TransformersRuntime,
    model_name: str,
    runtime: RuntimeConfig,
    teacher_mode: str,
    target_name: str,
    layer_ids: Optional[List[int]],
    rank: Optional[int],
    alpha: float,
) -> Dict[str, Any]:
    return {
        "transformers_version": transformers_runtime.version,
        "transformers_import_path": transformers_runtime.import_path,
        "model_name": model_name,
        "reasoning_mode": runtime.reasoning_mode,
        "teacher_mode": teacher_mode,
        "target_name": target_name,
        "layers": layer_ids,
        "rank": rank,
        "alpha": alpha,
        "max_new_tokens": resolve_generation_kwargs(model_name, runtime)["max_new_tokens"],
        "device": runtime.device,
        "requested_transformers_src": str(runtime.transformers_src.resolve()) if runtime.transformers_src else None,
    }


def fit_patch(
    tokenizer: Any,
    model: nn.Module,
    *,
    model_name: str,
    transformers_runtime: TransformersRuntime,
    docs_path: Path,
    corpus_dir: Path,
    benchmark_path: Path,
    out_path: Path,
    runtime: RuntimeConfig,
    layer_spec: str,
    target_name: str,
    rank: int,
    teacher_mode: str,
    alpha: float,
) -> Dict[str, Any]:
    layers = get_layers(model)
    layer_ids = resolve_layer_ids(len(layers), layer_spec)
    tools_backend = LocalTools(corpus_dir=corpus_dir, docs_path=docs_path)
    benchmark = read_jsonl(benchmark_path)
    train_items = [row for row in benchmark if row["split"] == "train_new"]

    u_bank: Dict[int, List[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}
    v_bank: Dict[int, List[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}
    trace_rows = []
    layer_shapes: Dict[str, List[int]] = {}

    for item in train_items:
        question = item["question"]
        if teacher_mode == "guided":
            teacher_run = run_guided_teacher(tokenizer, model, model_name, tools_backend, question, runtime)
        else:
            teacher_run = run_tool_agent(tokenizer, model, model_name, tools_backend, question, runtime)
        vanilla_run = run_vanilla(tokenizer, model, model_name, question, runtime)

        if not teacher_run.answer:
            raise RuntimeError(f"Teacher produced an empty answer for benchmark item {item['id']}")

        teacher_prefix = teacher_run.messages[:-1]
        vanilla_prefix = vanilla_run.messages[:-1]
        teacher_ids, teacher_pos = build_teacher_forcing_example(tokenizer, teacher_prefix, teacher_run.answer, tools=tool_specs())
        vanilla_ids, vanilla_pos = build_teacher_forcing_example(tokenizer, vanilla_prefix, teacher_run.answer, tools=None)

        teacher_acts = capture_linear_inputs(model, teacher_ids, layer_ids, target_name=target_name)
        vanilla_acts = capture_linear_inputs(model, vanilla_ids, layer_ids, target_name=target_name)

        for layer_id in layer_ids:
            module = get_target_module(get_layers(model)[layer_id], target_name)
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                layer_shapes[str(layer_id)] = list(module.weight.shape)
            for teacher_pos_idx, vanilla_pos_idx in zip(teacher_pos, vanilla_pos):
                teacher_act = teacher_acts[layer_id][teacher_pos_idx]
                vanilla_act = vanilla_acts[layer_id][vanilla_pos_idx]
                delta_a = teacher_act - vanilla_act
                denom = float(torch.dot(vanilla_act, vanilla_act).item()) + 1e-8
                u_bank[layer_id].append(linear_map_no_bias(module, delta_a))
                v_bank[layer_id].append(vanilla_act.float().cpu() / denom)

        trace_row = {
            "id": item["id"],
            "question": question,
            "teacher_answer": teacher_run.answer,
            "teacher_raw_answer": teacher_run.response.raw_text,
            "teacher_thinking": teacher_run.response.thinking,
            "teacher_tool_calls": teacher_run.response.tool_calls,
            "gold": item["answers"],
            "teacher_correct": answer_is_correct(teacher_run.answer, item["answers"]),
        }
        trace_rows.append(trace_row)
        print(f"traced {item['id']}: teacher_correct={trace_row['teacher_correct']} answer={teacher_run.answer!r}")

    patch = {
        "artifact": "patch",
        "metadata": build_artifact_metadata(
            transformers_runtime=transformers_runtime,
            model_name=model_name,
            runtime=runtime,
            teacher_mode=teacher_mode,
            target_name=target_name,
            layer_ids=layer_ids,
            rank=rank,
            alpha=alpha,
        ),
        "model_name": model_name,
        "target_name": target_name,
        "layer_ids": layer_ids,
        "layer_shapes": layer_shapes,
        "rank": rank,
        "teacher_mode": teacher_mode,
        "patches": {},
        "trace_rows": trace_rows,
    }
    for layer_id in layer_ids:
        left, right = compress_outer_sums(u_bank[layer_id], v_bank[layer_id], rank=rank)
        patch["patches"][str(layer_id)] = {"left": left, "right": right}
        print(f"layer {layer_id}: collected {len(u_bank[layer_id])} rank-1 factors -> rank-{rank} patch")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(patch, out_path)
    print(f"saved patch -> {out_path}")
    return patch


def validate_patch_compatibility(model: nn.Module, patch: Dict[str, Any], patch_path: Path) -> None:
    current_model_name = getattr(model.config, "_name_or_path", None) or getattr(model.config, "name_or_path", None)
    expected_model_name = patch.get("model_name")
    if expected_model_name and current_model_name and expected_model_name != current_model_name:
        raise RuntimeError(
            f"Patch {patch_path} was created for {expected_model_name}, but the current model is {current_model_name}."
        )

    target_name = patch["target_name"]
    layers = get_layers(model)
    for key, blob in patch["patches"].items():
        layer_id = int(key)
        if layer_id >= len(layers):
            raise RuntimeError(
                f"Patch {patch_path} targets layer {layer_id}, but the current model only has {len(layers)} layers."
            )
        base = get_target_module(layers[layer_id], target_name)
        if not hasattr(base, "weight") or not isinstance(base.weight, torch.Tensor):
            raise RuntimeError(f"Target module layer {layer_id}.{target_name} has no accessible weight tensor.")
        actual_shape = list(base.weight.shape)
        expected_shape = patch.get("layer_shapes", {}).get(str(layer_id))
        if expected_shape is not None and list(expected_shape) != actual_shape:
            raise RuntimeError(
                f"Patch {patch_path} expects layer {layer_id}.{target_name} shape {expected_shape}, "
                f"but found {actual_shape}."
            )
        left = blob["left"]
        right = blob["right"]
        if left.shape[0] != actual_shape[0] or right.shape[0] != actual_shape[1]:
            raise RuntimeError(
                f"Patch {patch_path} has incompatible factors for layer {layer_id}.{target_name}: "
                f"left={tuple(left.shape)} right={tuple(right.shape)} weight={tuple(actual_shape)}."
            )
        if left.shape[1] != right.shape[1]:
            raise RuntimeError(
                f"Patch {patch_path} has mismatched factor rank for layer {layer_id}: "
                f"left={tuple(left.shape)} right={tuple(right.shape)}."
            )


def apply_patch(model: nn.Module, patch_path: Path, alpha: float = 1.0) -> List[LowRankPatchedLinear]:
    patch = torch.load(patch_path, map_location="cpu")
    validate_patch_compatibility(model, patch, patch_path)
    target_name = patch["target_name"]
    layers = get_layers(model)
    wrappers: List[LowRankPatchedLinear] = []
    for key, blob in patch["patches"].items():
        layer_id = int(key)
        layer = layers[layer_id]
        if not hasattr(layer, "mlp"):
            raise RuntimeError("This patcher expects layer.mlp.<target> modules.")
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


def evaluate(
    tokenizer: Any,
    model: nn.Module,
    *,
    model_name: str,
    transformers_runtime: TransformersRuntime,
    docs_path: Path,
    corpus_dir: Path,
    benchmark_path: Path,
    out_path: Path,
    runtime: RuntimeConfig,
    mode: str,
    teacher_mode: str,
    target_name: str,
    layer_ids: Optional[List[int]],
    rank: Optional[int],
    alpha: float,
    report_mode: Optional[str] = None,
) -> Dict[str, Any]:
    tools_backend = LocalTools(corpus_dir=corpus_dir, docs_path=docs_path)
    benchmark = read_jsonl(benchmark_path)
    label = report_mode or mode
    rows = []

    for item in benchmark:
        if mode == "teacher":
            if teacher_mode == "guided":
                run = run_guided_teacher(tokenizer, model, model_name, tools_backend, item["question"], runtime)
            else:
                run = run_tool_agent(tokenizer, model, model_name, tools_backend, item["question"], runtime)
        else:
            run = run_vanilla(tokenizer, model, model_name, item["question"], runtime)

        correct = answer_is_correct(run.answer, item["answers"])
        row = {
            "id": item["id"],
            "split": item["split"],
            "question": item["question"],
            "pred": run.answer,
            "raw_pred": run.response.raw_text,
            "thinking": run.response.thinking,
            "tool_calls": run.response.tool_calls,
            "gold": item["answers"],
            "correct": correct,
        }
        rows.append(row)
        print(f"{label:>7s} | {item['id']} | correct={correct} | pred={run.answer!r}")

    scores = {}
    for split in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == split]
        scores[split] = sum(row["correct"] for row in subset) / max(1, len(subset))

    report = {
        "artifact": "evaluation",
        "metadata": build_artifact_metadata(
            transformers_runtime=transformers_runtime,
            model_name=model_name,
            runtime=runtime,
            teacher_mode=teacher_mode,
            target_name=target_name,
            layer_ids=layer_ids,
            rank=rank,
            alpha=alpha,
        ),
        "mode": label,
        "scores": scores,
        "rows": rows,
    }
    write_json(out_path, report)
    print(json.dumps(scores, indent=2))
    print(f"saved eval -> {out_path}")
    return report


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def build_sweep_summary(run_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_rows = []
    for entry in run_entries:
        splits = set()
        for report in entry["reports"].values():
            splits.update(report["scores"].keys())
        for split in sorted(splits):
            vanilla = entry["reports"]["vanilla"]["scores"].get(split)
            teacher = entry["reports"]["teacher"]["scores"].get(split)
            patched = entry["reports"]["patched"]["scores"].get(split)
            summary_rows.append(
                {
                    "model_name": entry["model_name"],
                    "reasoning_mode": entry["reasoning_mode"],
                    "split": split,
                    "vanilla": vanilla,
                    "teacher": teacher,
                    "patched": patched,
                    "teacher_minus_vanilla": None if teacher is None or vanilla is None else teacher - vanilla,
                    "patched_minus_vanilla": None if patched is None or vanilla is None else patched - vanilla,
                    "patch_path": str(entry["paths"]["patch"]),
                    "vanilla_report_path": str(entry["paths"]["vanilla"]),
                    "teacher_report_path": str(entry["paths"]["teacher"]),
                    "patched_report_path": str(entry["paths"]["patched"]),
                }
            )

    return {
        "artifact": "sweep_summary",
        "runs": run_entries,
        "rows": summary_rows,
    }


def write_sweep_summary(out_root: Path, summary: Dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "summary.json", summary)

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "reasoning_mode",
                "split",
                "vanilla",
                "teacher",
                "patched",
                "teacher_minus_vanilla",
                "patched_minus_vanilla",
                "patch_path",
                "vanilla_report_path",
                "teacher_report_path",
                "patched_report_path",
            ],
        )
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(row)

    md_lines = [
        "# Qwen3 Sweep Summary",
        "",
        "| Model | Reasoning | Split | Vanilla | Teacher | Patched | Teacher-Vanilla | Patched-Vanilla |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        md_lines.append(
            "| {model_name} | {reasoning_mode} | {split} | {vanilla} | {teacher} | {patched} | {teacher_delta} | {patched_delta} |".format(
                model_name=row["model_name"],
                reasoning_mode=row["reasoning_mode"],
                split=row["split"],
                vanilla=format_metric(row["vanilla"]),
                teacher=format_metric(row["teacher"]),
                patched=format_metric(row["patched"]),
                teacher_delta=format_metric(row["teacher_minus_vanilla"]),
                patched_delta=format_metric(row["patched_minus_vanilla"]),
            )
        )
    (out_root / "summary.md").write_text("\n".join(md_lines) + "\n")


def clear_model_memory(model: Optional[nn.Module]) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_corpus(args.corpus, args.docs, args.encoder)

    run_entries = []
    for model_name in parse_csv_arg(args.models):
        for reasoning_mode in parse_csv_arg(args.reasoning_modes):
            runtime = RuntimeConfig(
                reasoning_mode=reasoning_mode,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                transformers_src=args.transformers_src,
            )
            model_slug = slugify_model_name(model_name)
            run_dir = args.out_root / model_slug / reasoning_mode
            patch_path = run_dir / "patch.pt"
            vanilla_path = run_dir / "eval_vanilla.json"
            teacher_path = run_dir / "eval_teacher.json"
            patched_path = run_dir / "eval_patched.json"

            print(f"=== sweep: model={model_name} reasoning={reasoning_mode} teacher={args.teacher_mode} ===")
            tokenizer, model, transformers_runtime = load_tokenizer_and_model(
                model_name=model_name,
                runtime=runtime,
                load_in_4bit=args.load_in_4bit,
            )
            try:
                patch_blob = fit_patch(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name,
                    transformers_runtime=transformers_runtime,
                    docs_path=args.docs,
                    corpus_dir=args.corpus,
                    benchmark_path=args.benchmark,
                    out_path=patch_path,
                    runtime=runtime,
                    layer_spec=args.layers,
                    target_name=args.target,
                    rank=args.rank,
                    teacher_mode=args.teacher_mode,
                    alpha=args.alpha,
                )
                layer_ids = patch_blob["layer_ids"]

                vanilla_report = evaluate(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name,
                    transformers_runtime=transformers_runtime,
                    docs_path=args.docs,
                    corpus_dir=args.corpus,
                    benchmark_path=args.benchmark,
                    out_path=vanilla_path,
                    runtime=runtime,
                    mode="vanilla",
                    teacher_mode=args.teacher_mode,
                    target_name=args.target,
                    layer_ids=layer_ids,
                    rank=args.rank,
                    alpha=args.alpha,
                )
                teacher_report = evaluate(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name,
                    transformers_runtime=transformers_runtime,
                    docs_path=args.docs,
                    corpus_dir=args.corpus,
                    benchmark_path=args.benchmark,
                    out_path=teacher_path,
                    runtime=runtime,
                    mode="teacher",
                    teacher_mode=args.teacher_mode,
                    target_name=args.target,
                    layer_ids=layer_ids,
                    rank=args.rank,
                    alpha=args.alpha,
                )

                apply_patch(model, patch_path, alpha=args.alpha)
                patched_report = evaluate(
                    tokenizer=tokenizer,
                    model=model,
                    model_name=model_name,
                    transformers_runtime=transformers_runtime,
                    docs_path=args.docs,
                    corpus_dir=args.corpus,
                    benchmark_path=args.benchmark,
                    out_path=patched_path,
                    runtime=runtime,
                    mode="vanilla",
                    teacher_mode=args.teacher_mode,
                    target_name=args.target,
                    layer_ids=layer_ids,
                    rank=args.rank,
                    alpha=args.alpha,
                    report_mode="patched",
                )
            finally:
                clear_model_memory(model)

            run_entries.append(
                {
                    "model_name": model_name,
                    "reasoning_mode": reasoning_mode,
                    "paths": {
                        "patch": patch_path,
                        "vanilla": vanilla_path,
                        "teacher": teacher_path,
                        "patched": patched_path,
                    },
                    "reports": {
                        "vanilla": vanilla_report,
                        "teacher": teacher_report,
                        "patched": patched_report,
                    },
                }
            )

    summary = build_sweep_summary(run_entries)
    write_sweep_summary(args.out_root, summary)
    print(f"saved sweep summary -> {args.out_root}")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill tool-use traces into low-rank weight patches.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    build_parser = subparsers.add_parser("build-corpus")
    build_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH)
    build_parser.add_argument("--out", type=Path, default=DEFAULT_CORPUS_DIR)
    build_parser.add_argument("--encoder", default="tfidf")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--docs", type=Path, default=DEFAULT_DOCS_PATH)
    common.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    common.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    common.add_argument("--temperature", type=float, default=None)
    common.add_argument("--reasoning-mode", choices=QWEN3_REASONING_MODES, default=DEFAULT_REASONING_MODE)
    common.add_argument("--max-new-tokens", type=int, default=None)
    common.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    common.add_argument("--transformers-src", type=Path, default=DEFAULT_TRANSFORMERS_SRC)
    common.add_argument("--load-in-4bit", action="store_true")
    common.add_argument("--alpha", type=float, default=1.0)

    fit_parser = subparsers.add_parser("fit-patch", parents=[common])
    fit_parser.add_argument("--layers", default="-4,-3,-2,-1")
    fit_parser.add_argument("--target", default=DEFAULT_TARGET)
    fit_parser.add_argument("--rank", type=int, default=8)
    fit_parser.add_argument("--teacher-mode", choices=["guided", "auto"], default="guided")
    fit_parser.add_argument("--out", type=Path, default=DEFAULT_PATCH_PATH)

    eval_parser = subparsers.add_parser("evaluate", parents=[common])
    eval_parser.add_argument("--mode", choices=["vanilla", "teacher", "patched"], default="vanilla")
    eval_parser.add_argument("--teacher-mode", choices=["guided", "auto"], default="guided")
    eval_parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH_PATH)
    eval_parser.add_argument("--layers", default="-4,-3,-2,-1")
    eval_parser.add_argument("--target", default=DEFAULT_TARGET)
    eval_parser.add_argument("--rank", type=int, default=8)
    eval_parser.add_argument("--out", type=Path, default=DEFAULT_EVAL_PATH)

    sweep_parser = subparsers.add_parser("sweep", parents=[common])
    sweep_parser.add_argument("--models", default=",".join(DEFAULT_SWEEP_MODELS))
    sweep_parser.add_argument("--reasoning-modes", default="non_thinking,thinking")
    sweep_parser.add_argument("--teacher-mode", choices=["guided", "auto"], default="guided")
    sweep_parser.add_argument("--layers", default="-4,-3,-2,-1")
    sweep_parser.add_argument("--target", default=DEFAULT_TARGET)
    sweep_parser.add_argument("--rank", type=int, default=8)
    sweep_parser.add_argument("--encoder", default="tfidf")
    sweep_parser.add_argument("--out-root", type=Path, default=DEFAULT_SWEEP_ROOT)

    return parser


def runtime_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        reasoning_mode=args.reasoning_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        transformers_src=args.transformers_src,
    )


def main() -> None:
    args = make_parser().parse_args()

    if args.cmd == "build-corpus":
        build_corpus(args.docs, args.out, args.encoder)
        return

    if args.cmd == "sweep":
        run_sweep(args)
        return

    runtime = runtime_from_args(args)
    tokenizer, model, transformers_runtime = load_tokenizer_and_model(
        model_name=args.model,
        runtime=runtime,
        load_in_4bit=args.load_in_4bit,
    )

    try:
        if args.cmd == "fit-patch":
            fit_patch(
                tokenizer=tokenizer,
                model=model,
                model_name=args.model,
                transformers_runtime=transformers_runtime,
                docs_path=args.docs,
                corpus_dir=args.corpus,
                benchmark_path=args.benchmark,
                out_path=args.out,
                runtime=runtime,
                layer_spec=args.layers,
                target_name=args.target,
                rank=args.rank,
                teacher_mode=args.teacher_mode,
                alpha=args.alpha,
            )
            return

        if args.cmd == "evaluate":
            layer_ids = resolve_layer_ids(len(get_layers(model)), args.layers)
            run_mode = args.mode
            report_mode = args.mode
            if args.mode == "patched":
                apply_patch(model, args.patch, alpha=args.alpha)
                run_mode = "vanilla"
                report_mode = "patched"

            evaluate(
                tokenizer=tokenizer,
                model=model,
                model_name=args.model,
                transformers_runtime=transformers_runtime,
                docs_path=args.docs,
                corpus_dir=args.corpus,
                benchmark_path=args.benchmark,
                out_path=args.out,
                runtime=runtime,
                mode=run_mode,
                teacher_mode=args.teacher_mode,
                target_name=args.target,
                layer_ids=layer_ids,
                rank=args.rank,
                alpha=args.alpha,
                report_mode=report_mode,
            )
            return
    finally:
        clear_model_memory(model)


if __name__ == "__main__":
    main()
