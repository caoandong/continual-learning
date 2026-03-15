from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import qwen_thought_patch_cli as cli


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


def test_render_table_contains_headers_and_rows():
    table = cli.render_table(["A", "B"], [[1, 2], [3, 4]])
    assert "A" in table
    assert "B" in table
    assert "1" in table
    assert "4" in table


def test_find_subsequence_positions_finds_match():
    positions = cli.find_subsequence_positions([9, 1, 2, 3, 7], [2, 3])
    assert positions == [2, 3]
