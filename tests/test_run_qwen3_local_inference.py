from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


def load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen3_local_inference.py"
    spec = importlib.util.spec_from_file_location("run_qwen3_local_inference", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = load_script_module()


def test_build_cases_defaults():
    cases = script.build_cases(None)

    assert len(cases) == 2
    assert cases[0].expected_substring == "4"
    assert cases[1].expected_substring == "paris"


def test_build_cases_from_custom_prompts():
    cases = script.build_cases(["One", "Two"])

    assert [case.prompt for case in cases] == ["One", "Two"]
    assert all(case.expected_substring is None for case in cases)


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(script.torch.backends, "mps", None, raising=False)

    assert script.resolve_device("auto") == "cuda"


def test_resolve_device_raises_for_missing_cuda(monkeypatch):
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        script.resolve_device("cuda")


def test_model_kwargs_for_cuda_uses_bfloat16():
    kwargs = script.model_kwargs_for_device("cuda")

    assert kwargs["dtype"] is torch.bfloat16
    assert kwargs["device_map"] == {"": 0}


def test_assert_expected_is_case_insensitive():
    case = script.PromptCase(prompt="Capital?", expected_substring="paris")

    script.assert_expected(case, "Paris")


def test_assert_expected_raises_on_missing_substring():
    case = script.PromptCase(prompt="Math?", expected_substring="4")

    with pytest.raises(RuntimeError, match="Smoke test failed"):
        script.assert_expected(case, "five")
