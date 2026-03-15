import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qwen_thought_patch_cli as cli


LOCAL_QWEN_CHECKPOINT_ROOT = Path("/content/drive/MyDrive/flair/software/qwen3/checkpoints")


def test_sample_examples_returns_unique_triples_and_correct_answers():
    dataset = cli.sample_examples(
        "multiply",
        train_examples=10,
        eval_examples=20,
        seed=0,
        digit_min=1,
        digit_max=9,
    )

    all_examples = dataset["train"] + dataset["eval"]
    triples = [example.numbers for example in all_examples]
    assert len(triples) == len(set(triples))
    assert all(example.answer == example.numbers[0] * example.numbers[1] * example.numbers[2] for example in all_examples)


def test_solve_weight_update_matches_identity_design():
    src = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    update = cli.solve_weight_update(src, target, rho=0.0)

    assert update.shape == (2, 2)
    assert torch.allclose(src @ update.T, target)


def test_normalize_prediction_prefers_last_integer():
    assert cli.normalize_prediction("3, 4, 7 -> 84") == "84"
    assert cli.normalize_prediction("The answer is 14.") == "14"


def test_contains_expected_integer_uses_digit_boundaries():
    assert cli.contains_expected_integer("The answer is 84.", "84")
    assert not cli.contains_expected_integer("The answer is 184.", "84")


def test_compute_fc3_target_absorbs_dz_without_bias():
    ctx_mlp_out = torch.tensor([[1.0, 2.0]])
    raw_mlp_out = torch.tensor([[0.25, 0.5]])
    dz = torch.tensor([[0.75, -0.25]])

    no_bias_target = cli.compute_fc3_target(
        ctx_mlp_out=ctx_mlp_out,
        raw_mlp_out=raw_mlp_out,
        dz=dz,
        use_output_bias=False,
    )
    bias_target = cli.compute_fc3_target(
        ctx_mlp_out=ctx_mlp_out,
        raw_mlp_out=raw_mlp_out,
        dz=dz,
        use_output_bias=True,
    )

    assert torch.allclose(no_bias_target, torch.tensor([[1.5, 1.25]]))
    assert torch.allclose(bias_target, torch.tensor([[0.75, 1.5]]))


def test_render_table_contains_headers_and_rows():
    table = cli.render_table(["A", "B"], [[1, 2], [3, 4]])
    assert "A" in table
    assert "B" in table
    assert "1" in table
    assert "4" in table


def test_find_subsequence_positions_finds_match():
    positions = cli.find_subsequence_positions([9, 1, 2, 3, 7], [2, 3])
    assert positions == [2, 3]


def test_resolve_local_checkpoint_dir_prefers_matching_child(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "Qwen3-0.6B"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "tokenizer.json").write_text("{}")
    (checkpoint_dir / "model.safetensors").write_text("stub")

    resolved = cli.resolve_local_checkpoint_dir(
        model_size="0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        local_dir=str(checkpoint_root),
    )

    assert resolved == str(checkpoint_dir)


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.backends, "mps", None, raising=False)

    assert cli.resolve_device("auto") == "cuda"


def test_resolve_device_raises_for_missing_cuda(monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        cli.resolve_device("cuda")


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, chat_wrapped=None):
        if chat_wrapped:
            return [10, 11]
        return [int(text)]


class FakeModel(torch.nn.Module):
    def forward(self, input_ids, thought_patches=None):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 128)
        logits[:, 1, 6] = 10.0
        return logits


def test_evaluate_teacher_forced_examples_scores_answer_tokens():
    metrics = cli.evaluate_teacher_forced_examples(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        thought_patches=None,
        examples=[cli.ArithmeticExample(numbers=(1, 2, 3), answer=6)],
        task=cli.TASKS["sum"],
        device="cpu",
        mode="raw",
    )

    assert metrics["sequence_accuracy"] == pytest.approx(100.0)
    assert metrics["token_accuracy"] == pytest.approx(100.0)
    assert metrics["avg_logprob"] > -0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the local Qwen E2E smoke test.")
def test_qwen_thought_patch_cli_smoke_e2e_uses_local_gpu_checkpoint(tmp_path: Path):
    if not LOCAL_QWEN_CHECKPOINT_ROOT.exists():
        pytest.skip(f"Local checkpoint root is missing: {LOCAL_QWEN_CHECKPOINT_ROOT}")

    resolved_dir = cli.resolve_local_checkpoint_dir(
        model_size="0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        local_dir=str(LOCAL_QWEN_CHECKPOINT_ROOT),
    )
    if not cli.checkpoint_dir_has_files(Path(resolved_dir)):
        pytest.skip(f"Local checkpoint files are incomplete under: {resolved_dir}")

    out_path = tmp_path / "qwen_thought_patch_smoke.json"
    command = [
        sys.executable,
        "qwen_thought_patch_cli.py",
        "--model-size",
        "0.6B",
        "--model-type",
        "instruct",
        "--local-dir",
        str(LOCAL_QWEN_CHECKPOINT_ROOT),
        "--device",
        "cuda",
        "--tasks",
        "multiply",
        "--train-examples",
        "1",
        "--eval-examples",
        "1",
        "--seeds",
        "1",
        "--learning-rate",
        "0.1",
        "--rho",
        "0.0",
        "--max-new-tokens",
        "8",
        "--fit-mode",
        "sequential",
        "--no-paper-filter",
        "--no-step-eval",
        "--log-level",
        "WARNING",
        "--out",
        str(out_path),
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    artifact = json.loads(out_path.read_text())
    assert artifact["config"]["device"] == "cuda"
    assert artifact["config"]["local_dir"] == resolved_dir
    assert artifact["config"]["tasks"] == ["multiply"]
    assert len(artifact["results"]) == 1
    result = artifact["results"][0]
    assert result["task"] == "multiply"
    assert "final_patched_eval" in result
    assert "Paper-style summary" in completed.stdout
