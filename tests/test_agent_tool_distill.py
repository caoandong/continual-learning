from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_tool_distill as distill


class DummyTokenizer:
    def __init__(self):
        self.response_schema = distill.QWEN3_RESPONSE_SCHEMA
        self.eos_token = "</s>"
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.tokenized_texts = []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
        rendered = "\n".join(f"{message['role']}:{message.get('content', '')}" for message in messages)
        if tools:
            rendered += "\nTOOLS"
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        self.tokenized_texts.append(text)
        tokens = [max(1, (ord(ch) % 17) + 1) for ch in text]
        return SimpleNamespace(input_ids=torch.tensor([tokens], dtype=torch.long))


class TinyMLP(nn.Module):
    def __init__(self, in_features=3, out_features=5):
        super().__init__()
        self.up_proj = nn.Linear(in_features, out_features, bias=False)


class TinyLayer(nn.Module):
    def __init__(self, in_features=3, out_features=5):
        super().__init__()
        self.mlp = TinyMLP(in_features=in_features, out_features=out_features)


class TinyBackbone(nn.Module):
    def __init__(self, in_features=3, out_features=5):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(in_features=in_features, out_features=out_features)])


class TinyModel(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-1.7B", in_features=3, out_features=5):
        super().__init__()
        self.model = TinyBackbone(in_features=in_features, out_features=out_features)
        self.config = SimpleNamespace(_name_or_path=model_name)


def test_parse_assistant_response_qwen3_tool_calls_match_local_schema():
    tokenizer = DummyTokenizer()
    raw = (
        "<tool_call>\n"
        "<function=search_news>\n"
        "<parameter=query>\n"
        "\"qwen3 launch\"\n"
        "</parameter>\n"
        "<parameter=top_k>\n"
        "4\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    response = distill.parse_assistant_response(tokenizer, raw, "Qwen/Qwen3-1.7B")

    assert response.tool_calls == [
        distill.make_tool_call(
            "search_news",
            {
                "query": "qwen3 launch",
                "top_k": 4,
            },
        )
    ]
    assert response.content == ""
    assert response.thinking is None


def test_parse_assistant_response_qwen3_thinking_keeps_final_content():
    tokenizer = DummyTokenizer()
    raw = "<think>\nI should inspect the retrieved snippets.\n</think>\nThe answer is Qwen3."
    response = distill.parse_assistant_response(tokenizer, raw, "Qwen/Qwen3-1.7B")

    assert response.content == "The answer is Qwen3."
    assert response.thinking == "I should inspect the retrieved snippets."
    assert response.tool_calls == []


def test_guided_teacher_serializes_tool_history(monkeypatch):
    tokenizer = DummyTokenizer()
    runtime = distill.RuntimeConfig(reasoning_mode="non_thinking")

    def fake_generate(**_kwargs):
        return distill.AssistantResponse(
            raw_text="<think>hidden</think>Final answer.",
            content="Final answer.",
            thinking="hidden",
            tool_calls=[],
        )

    monkeypatch.setattr(distill, "generate_from_messages", fake_generate)
    tools_backend = SimpleNamespace(search_news=lambda query, top_k=4: json.dumps({"query": query, "top_k": top_k}))

    run = distill.run_guided_teacher(
        tokenizer=tokenizer,
        model=object(),
        model_name="Qwen/Qwen3-1.7B",
        tools_backend=tools_backend,
        question="What launched?",
        runtime=runtime,
    )

    assert run.messages[2]["tool_calls"][0]["function"]["name"] == "search_news"
    assert run.messages[2]["tool_calls"][0]["function"]["arguments"] == {"query": "What launched?", "top_k": 4}
    assert run.messages[3]["role"] == "tool"
    assert run.messages[-1] == {"role": "assistant", "content": "Final answer."}
    assert run.answer == "Final answer."


def test_teacher_forcing_replay_uses_final_content_only(monkeypatch):
    tokenizer = DummyTokenizer()
    runtime = distill.RuntimeConfig(reasoning_mode="thinking")

    def fake_generate(**_kwargs):
        return distill.AssistantResponse(
            raw_text="<think>scratchpad</think>Visible answer.",
            content="Visible answer.",
            thinking="scratchpad",
            tool_calls=[],
        )

    monkeypatch.setattr(distill, "generate_from_messages", fake_generate)
    run = distill.run_vanilla(
        tokenizer=tokenizer,
        model=object(),
        model_name="Qwen/Qwen3-1.7B",
        question="Question?",
        runtime=runtime,
    )
    _, positions = distill.build_teacher_forcing_example(
        tokenizer=tokenizer,
        prefix_messages=run.messages[:-1],
        answer_text_value=run.answer,
        tools=None,
    )

    assert run.answer == "Visible answer."
    assert run.messages[-1]["content"] == "Visible answer."
    assert positions
    assert all("<think>" not in text for text in tokenizer.tokenized_texts)


def test_apply_patch_rejects_model_name_mismatch(tmp_path):
    model = TinyModel(model_name="Qwen/Qwen3-1.7B")
    patch_path = tmp_path / "patch.pt"
    torch.save(
        {
            "model_name": "Qwen/Qwen3-4B",
            "target_name": "up_proj",
            "layer_ids": [0],
            "layer_shapes": {"0": [5, 3]},
            "patches": {"0": {"left": torch.zeros(5, 2), "right": torch.zeros(3, 2)}},
        },
        patch_path,
    )

    with pytest.raises(RuntimeError, match="created for Qwen/Qwen3-4B"):
        distill.apply_patch(model, patch_path)


def test_apply_patch_rejects_factor_shape_mismatch(tmp_path):
    model = TinyModel(model_name="Qwen/Qwen3-1.7B")
    patch_path = tmp_path / "patch.pt"
    torch.save(
        {
            "model_name": "Qwen/Qwen3-1.7B",
            "target_name": "up_proj",
            "layer_ids": [0],
            "layer_shapes": {"0": [5, 3]},
            "patches": {"0": {"left": torch.zeros(6, 2), "right": torch.zeros(3, 2)}},
        },
        patch_path,
    )

    with pytest.raises(RuntimeError, match="incompatible factors"):
        distill.apply_patch(model, patch_path)


def test_cli_help_includes_sweep_and_new_runtime_flags():
    parser = distill.make_parser()
    help_text = parser.format_help()
    subparsers = next(action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction")
    fit_help = subparsers.choices["fit-patch"].format_help()

    assert "sweep" in help_text
    assert "--reasoning-mode" in fit_help
    assert "--transformers-src" in fit_help


def test_qwen3_generation_profiles_are_explicit():
    thinking = distill.resolve_generation_kwargs(
        "Qwen/Qwen3-1.7B",
        distill.RuntimeConfig(reasoning_mode="thinking"),
    )
    non_thinking = distill.resolve_generation_kwargs(
        "Qwen/Qwen3-1.7B",
        distill.RuntimeConfig(reasoning_mode="non_thinking"),
    )

    assert thinking["enable_thinking"] is True
    assert thinking["max_new_tokens"] == 512
    assert non_thinking["enable_thinking"] is False
    assert non_thinking["temperature"] == pytest.approx(0.7)


def test_run_sweep_writes_aggregate_reports(monkeypatch, tmp_path):
    parser = distill.make_parser()
    args = parser.parse_args(
        [
            "sweep",
            "--models",
            "Qwen/Qwen3-0.6B",
            "--reasoning-modes",
            "non_thinking,thinking",
            "--out-root",
            str(tmp_path / "runs"),
        ]
    )

    monkeypatch.setattr(distill, "ensure_corpus", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        distill,
        "load_tokenizer_and_model",
        lambda model_name, runtime, load_in_4bit=False: (
            DummyTokenizer(),
            object(),
            distill.TransformersRuntime(version="4.57.6", import_path="/tmp/transformers/__init__.py"),
        ),
    )

    def fake_fit_patch(**kwargs):
        kwargs["out_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": kwargs["model_name"],
                "target_name": kwargs["target_name"],
                "layer_ids": [0],
                "layer_shapes": {"0": [5, 3]},
                "patches": {"0": {"left": torch.zeros(5, 1), "right": torch.zeros(3, 1)}},
            },
            kwargs["out_path"],
        )
        return {"layer_ids": [0]}

    def fake_evaluate(*, out_path, mode, report_mode=None, **kwargs):
        label = report_mode or mode
        score_map = {
            "vanilla": {"train_new": 0.25, "eval_new": 0.5, "eval_old": 0.8},
            "teacher": {"train_new": 0.75, "eval_new": 0.875, "eval_old": 0.8},
            "patched": {"train_new": 0.5, "eval_new": 0.75, "eval_old": 0.8},
        }
        report = {"mode": label, "scores": score_map[label], "rows": []}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report))
        return report

    monkeypatch.setattr(distill, "fit_patch", fake_fit_patch)
    monkeypatch.setattr(distill, "evaluate", fake_evaluate)
    monkeypatch.setattr(distill, "apply_patch", lambda *args, **kwargs: [])
    monkeypatch.setattr(distill, "clear_model_memory", lambda model: None)

    summary = distill.run_sweep(args)

    assert (args.out_root / "summary.json").exists()
    assert (args.out_root / "summary.csv").exists()
    assert (args.out_root / "summary.md").exists()
    assert len(summary["rows"]) == 6
    train_new_rows = [row for row in summary["rows"] if row["split"] == "train_new"]
    eval_new_rows = [row for row in summary["rows"] if row["split"] == "eval_new"]
    assert train_new_rows[0]["teacher_minus_vanilla"] == pytest.approx(0.5)
    assert train_new_rows[0]["patched_minus_vanilla"] == pytest.approx(0.25)
    assert eval_new_rows[0]["teacher_minus_vanilla"] == pytest.approx(0.375)
