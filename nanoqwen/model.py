from dataclasses import dataclass
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

QWEN3_BASE_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

QWEN3_CONFIGS = {
    "0.6B":  {**QWEN3_BASE_CONFIG, "emb_dim": 1024, "n_heads": 16, "n_layers": 28, "hidden_dim": 3072},
    "1.7B":  {**QWEN3_BASE_CONFIG, "emb_dim": 2048, "n_heads": 16, "n_layers": 28, "hidden_dim": 6144},
    "4B":    {**QWEN3_BASE_CONFIG, "emb_dim": 2560, "n_heads": 32, "n_layers": 36, "hidden_dim": 9728},
    "8B":    {**QWEN3_BASE_CONFIG, "emb_dim": 4096, "n_heads": 32, "n_layers": 36, "hidden_dim": 12288},
    "14B":   {**QWEN3_BASE_CONFIG, "emb_dim": 5120, "n_heads": 40, "n_layers": 40, "hidden_dim": 17408},
    "32B":   {**QWEN3_BASE_CONFIG, "emb_dim": 5120, "n_heads": 64, "n_layers": 64, "hidden_dim": 25600},
}


@dataclass
class ThoughtPatch:
    d_fc1: torch.Tensor
    d_fc2: torch.Tensor
    d_fc3: torch.Tensor
    d_bias: Optional[torch.Tensor] = None

    def zero_(self):
        self.d_fc1.zero_()
        self.d_fc2.zero_()
        self.d_fc3.zero_()
        if self.d_bias is not None:
            self.d_bias.zero_()
        return self

    def to(self, device=None, dtype=None):
        return ThoughtPatch(
            d_fc1=self.d_fc1.to(device=device, dtype=dtype),
            d_fc2=self.d_fc2.to(device=device, dtype=dtype),
            d_fc3=self.d_fc3.to(device=device, dtype=dtype),
            d_bias=None if self.d_bias is None else self.d_bias.to(device=device, dtype=dtype),
        )

    def norms(self):
        return {
            "fc1": float(self.d_fc1.float().norm().item()),
            "fc2": float(self.d_fc2.float().norm().item()),
            "fc3": float(self.d_fc3.float().norm().item()),
            "bias": 0.0 if self.d_bias is None else float(self.d_bias.float().norm().item()),
        }


def build_empty_thought_patches(model):
    patches = []
    for block in model.trf_blocks:
        ff = block.ff
        patches.append(
            ThoughtPatch(
                d_fc1=torch.zeros_like(ff.fc1.weight),
                d_fc2=torch.zeros_like(ff.fc2.weight),
                d_fc3=torch.zeros_like(ff.fc3.weight),
                d_bias=torch.zeros(ff.fc3.weight.shape[0], device=ff.fc3.weight.device, dtype=ff.fc3.weight.dtype),
            )
        )
    return patches


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x, patch=None, return_trace=False):
        fc1_weight = self.fc1.weight
        fc2_weight = self.fc2.weight
        fc3_weight = self.fc3.weight
        fc3_bias = None

        if patch is not None:
            fc1_weight = fc1_weight + patch.d_fc1.to(device=fc1_weight.device, dtype=fc1_weight.dtype)
            fc2_weight = fc2_weight + patch.d_fc2.to(device=fc2_weight.device, dtype=fc2_weight.dtype)
            fc3_weight = fc3_weight + patch.d_fc3.to(device=fc3_weight.device, dtype=fc3_weight.dtype)
            if patch.d_bias is not None:
                fc3_bias = patch.d_bias.to(device=fc3_weight.device, dtype=fc3_weight.dtype)

        x_fc1 = nn.functional.linear(x, fc1_weight, None)
        x_fc2 = nn.functional.linear(x, fc2_weight, None)
        hidden = nn.functional.silu(x_fc1) * x_fc2
        out = nn.functional.linear(hidden, fc3_weight, None)

        if fc3_bias is not None:
            out = out + fc3_bias

        if return_trace:
            return out, {
                "mlp_in": x,
                "fc1_out": x_fc1,
                "fc2_out": x_fc2,
                "mlp_hidden": hidden,
                "mlp_out": out,
            }

        return out


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, patch=None, return_trace=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + shortcut
        att_resid = x

        shortcut = x
        mlp_in = self.norm2(x)
        if return_trace:
            mlp_out, ff_trace = self.ff(mlp_in, patch=patch, return_trace=True)
        else:
            mlp_out = self.ff(mlp_in, patch=patch, return_trace=False)
        x = mlp_out + shortcut

        if return_trace:
            return x, {
                "att_resid": att_resid,
                "mlp_in": ff_trace["mlp_in"],
                "fc1_out": ff_trace["fc1_out"],
                "fc2_out": ff_trace["fc2_out"],
                "mlp_hidden": ff_trace["mlp_hidden"],
                "mlp_out": ff_trace["mlp_out"],
                "resid_out": x,
            }

        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        head_dim = cfg["head_dim"] if cfg["head_dim"] is not None else cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx, thought_patches: Optional[Sequence[ThoughtPatch]] = None, return_trace=False):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        if thought_patches is not None and len(thought_patches) != len(self.trf_blocks):
            raise ValueError(
                f"Expected {len(self.trf_blocks)} thought patches, received {len(thought_patches)}."
            )

        traces = [] if return_trace else None

        for layer_id, block in enumerate(self.trf_blocks):
            patch = None if thought_patches is None else thought_patches[layer_id]
            if return_trace:
                x, block_trace = block(x, mask, self.cos, self.sin, patch=patch, return_trace=True)
                traces.append(block_trace)
            else:
                x = block(x, mask, self.cos, self.sin, patch=patch, return_trace=False)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        if return_trace:
            return logits, traces

        return logits


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>"
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                 apply_chat_template=True, add_generation_prompt=False, add_thinking=False):

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid

        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight


def load_tokenizer(model_size="0.6B", model_type="base", repo_id=None, local_dir=None):
    """Download and create a Qwen3Tokenizer.

    Args:
        model_size: One of "0.6B", "1.7B", "4B", "8B", "14B", "32B".
        model_type: One of "base", "instruct", "reasoning".
        repo_id: HuggingFace repo id. Derived from model_size/model_type if None.
        local_dir: Local directory for downloaded files. Defaults to "Qwen3-{model_size}[-Base]".
    """
    is_base = model_type == "base"

    if repo_id is None:
        repo_id = f"Qwen/Qwen3-{model_size}" if not is_base else f"Qwen/Qwen3-{model_size}-Base"

    if local_dir is None:
        local_dir = f"Qwen3-{model_size}" if not is_base else f"Qwen3-{model_size}-Base"

    local_dir = Path(local_dir)
    tokenizer_file_path = local_dir / "tokenizer.json"
    if not tokenizer_file_path.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            local_dir=str(local_dir),
        )

    use_chat = model_type in ("instruct", "reasoning")

    return Qwen3Tokenizer(
        tokenizer_file_path=str(tokenizer_file_path),
        repo_id=repo_id,
        apply_chat_template=use_chat,
        add_generation_prompt=use_chat,
        add_thinking=(model_type == "reasoning"),
    )


def download_weights(repo_id, local_dir):
    """Download safetensors weights from HuggingFace and return a flat state dict."""
    local_dir = Path(local_dir)
    single_file = local_dir / "model.safetensors"
    index_file = local_dir / "model.safetensors.index.json"

    if single_file.exists():
        return load_file(str(single_file))

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)

        weights = {}
        for filename in set(index["weight_map"].values()):
            shard = load_file(str(local_dir / filename))
            weights.update(shard)
        return weights

    # Try single-file first (small models like 0.6B)
    try:
        hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=str(local_dir))
        return load_file(str(single_file))
    except Exception:
        pass

    # Fall back to sharded weights
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    with open(os.path.join(repo_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)

    weights = {}
    for filename in set(index["weight_map"].values()):
        shard = load_file(os.path.join(repo_dir, filename))
        weights.update(shard)
    return weights


def load_model_and_tokenizer(model_size="0.6B", model_type="base", device="cpu",
                             repo_id=None, local_dir=None, dtype=None):
    """Download weights + tokenizer, build and return (model, tokenizer).

    Args:
        model_size: One of "0.6B", "1.7B", "4B", "8B", "14B", "32B".
        model_type: One of "base", "instruct", "reasoning".
        device: Target device for the model (e.g. "cpu", "cuda").
        repo_id: HuggingFace repo id. Derived from model_size/model_type if None.
        local_dir: Local cache directory. Derived from repo_id if None.
    """
    is_base = model_type == "base"

    if repo_id is None:
        repo_id = f"Qwen/Qwen3-{model_size}-Base" if is_base else f"Qwen/Qwen3-{model_size}"

    if local_dir is None:
        local_dir = Path(repo_id).parts[-1]

    cfg = dict(QWEN3_CONFIGS[model_size])
    if dtype is not None:
        cfg["dtype"] = dtype

    # Build model and load weights
    model = Qwen3Model(cfg)
    weights = download_weights(repo_id, local_dir)
    load_weights_into_qwen(model, cfg, weights)
    del weights
    model.to(device)

    # Load tokenizer
    tokenizer = load_tokenizer(
        model_size=model_size, model_type=model_type,
        repo_id=repo_id, local_dir=local_dir,
    )

    return model, tokenizer


def generate_text_simple(model, token_ids, max_new_tokens=500, eos_token_id=None, thought_patches=None):
    """Yield one token tensor at a time (greedy, no sampling)."""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids, thought_patches=thought_patches)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


def generate_and_print(model, tokenizer, input_token_ids, device="cpu", max_new_tokens=500, thought_patches=None):
    """Run greedy generation, print tokens as they arrive, and report speed."""
    input_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    generated_tokens = 0

    for token in generate_text_simple(
        model=model,
        token_ids=input_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        thought_patches=thought_patches,
    ):
        generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print(f"\n\nGeneration speed: {tokens_per_sec:.2f} tokens/sec")

    if torch.cuda.is_available():
        gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        print(f"GPU memory used: {gb:.2f} GB")
