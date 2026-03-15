import copy
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanoqwen.model import Qwen3Model, build_empty_thought_patches


def tiny_cfg():
    return {
        "vocab_size": 64,
        "context_length": 32,
        "head_dim": 8,
        "qk_norm": False,
        "n_kv_groups": 2,
        "rope_base": 10_000.0,
        "dtype": torch.float32,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "hidden_dim": 24,
    }


def test_build_empty_thought_patches_match_model_shapes():
    model = Qwen3Model(tiny_cfg())
    patches = build_empty_thought_patches(model)

    assert len(patches) == len(model.trf_blocks)
    for patch, block in zip(patches, model.trf_blocks):
        assert patch.d_fc1.shape == block.ff.fc1.weight.shape
        assert patch.d_fc2.shape == block.ff.fc2.weight.shape
        assert patch.d_fc3.shape == block.ff.fc3.weight.shape
        assert patch.d_bias is None


def test_build_empty_thought_patches_can_enable_debug_bias():
    model = Qwen3Model(tiny_cfg())
    patches = build_empty_thought_patches(model, include_bias=True)

    for patch, block in zip(patches, model.trf_blocks):
        assert patch.d_bias is not None
        assert patch.d_bias.shape == (block.ff.fc3.weight.shape[0],)


def test_forward_return_trace_exposes_expected_keys():
    model = Qwen3Model(tiny_cfg())
    input_ids = torch.randint(0, 64, (1, 5))

    logits, traces = model(input_ids, return_trace=True)

    assert logits.shape == (1, 5, 64)
    assert len(traces) == 2
    for trace in traces:
        assert set(trace) >= {"att_resid", "mlp_in", "mlp_hidden", "mlp_out", "resid_out"}


def test_nonzero_patch_changes_logits():
    model = Qwen3Model(tiny_cfg())
    input_ids = torch.randint(0, 64, (1, 5))
    patches = build_empty_thought_patches(model)
    patches[0].d_fc3[0, 0] += 0.25

    base_logits = model(input_ids)
    patched_logits = model(input_ids, thought_patches=patches)

    assert not torch.allclose(base_logits, patched_logits)


def test_debug_bias_patch_changes_logits():
    model = Qwen3Model(tiny_cfg())
    input_ids = torch.randint(0, 64, (1, 5))
    patches = build_empty_thought_patches(model, include_bias=True)
    patches[0].d_bias += 0.25

    base_logits = model(input_ids)
    patched_logits = model(input_ids, thought_patches=patches)

    assert not torch.allclose(base_logits, patched_logits)


def test_zero_patch_is_identity():
    model = Qwen3Model(tiny_cfg())
    input_ids = torch.randint(0, 64, (1, 5))
    zero_patches = build_empty_thought_patches(model)
    cloned = [copy.deepcopy(patch) for patch in zero_patches]

    base_logits = model(input_ids)
    patched_logits = model(input_ids, thought_patches=cloned)

    assert torch.allclose(base_logits, patched_logits)
