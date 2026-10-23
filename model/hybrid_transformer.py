"""
CodeForge-H: Qwen3.8-Style Hybrid-Attention Reasoning Decoder LLM.
Implements Section K and Section N specifications:
- Repeating groups of [DeltaNet x3, Attention x1] across 12 layers (PILOT tier: 125M-150M parameters)
- Residual blocks: x = x + Mixer(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x))
- Attention with 12 Q heads, 4 KV heads (GQA), head_dim 64, RoPE (base 10,000), sigmoid output gate
- DeltaNet linear attention with 6 heads, head_dim 128, value expansion 2, short conv kernel 4
- Tied LM head embeddings
- Multi-Token Prediction (MTP) draft head for speculative decoding
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.deltanet_reference import DeltaNetReference

@dataclass
class CodeForgeHConfig:
    vocab_size: int = 49152
    d_model: int = 768
    n_layers: int = 12             # 3 groups of [DeltaNet x3, Attention x1]
    ffn_dim: int = 1792            # Tuned to achieve 135M-145M target parameters
    attn_heads: int = 12
    attn_kv_heads: int = 4
    attn_head_dim: int = 64
    deltanet_heads: int = 6
    deltanet_head_dim: int = 128
    deltanet_value_expansion: int = 2
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    mtp_aux_weight: float = 0.2
    dropout: float = 0.0

    def is_attention_layer(self, layer_idx: int) -> bool:
        """Every 4th layer is full Multi-Head Causal Attention: [D, D, D, A, D, D, D, A, ...]"""
        return (layer_idx + 1) % 4 == 0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

class GatedCausalAttention(nn.Module):
    """
    Attention layer with GQA, RoPE, and Sigmoid Output Gating per Section K.
    """
    def __init__(self, config: CodeForgeHConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.attn_heads
        self.n_kv_heads = config.attn_kv_heads
        self.head_dim = config.attn_head_dim
        self.num_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.attn_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.attn_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.attn_kv_heads * self.head_dim, bias=False)

        # Sigmoid output gate projection
        self.gate_proj = nn.Linear(config.d_model, config.attn_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.attn_heads * self.head_dim, config.d_model, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        q = self.q_proj(x).view(b, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, seq_len, self.n_kv_heads, self.head_dim)
        gate = torch.sigmoid(self.gate_proj(x))

        cos, sin = self.rotary(x, seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.num_rep > 1:
            k = k.repeat_interleave(self.num_rep, dim=2)
            v = v.repeat_interleave(self.num_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        out = out * gate  # Sigmoid output gate
        return self.o_proj(out)

class SwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)  # Gate
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)  # Down
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class HybridBlock(nn.Module):
    def __init__(self, layer_idx: int, config: CodeForgeHConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attn = config.is_attention_layer(layer_idx)

        self.mixer_norm = RMSNorm(config.d_model)
        if self.is_attn:
            self.mixer = GatedCausalAttention(config)
        else:
            self.mixer = DeltaNetReference(
                d_model=config.d_model,
                num_heads=config.deltanet_heads,
                head_dim=config.deltanet_head_dim,
                value_expansion=config.deltanet_value_expansion
            )

        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUBlock(config.d_model, config.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Residual: x = x + Mixer(RMSNorm(x))
        if self.is_attn:
            x = x + self.mixer(self.mixer_norm(x))
        else:
            mixer_out, _ = self.mixer(self.mixer_norm(x))
            x = x + mixer_out
        # Residual FFN: x = x + SwiGLU(RMSNorm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class CodeForgeHModel(nn.Module):
    def __init__(self, config: Optional[CodeForgeHConfig] = None):
        super().__init__()
        self.config = config or CodeForgeHConfig()

        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.layers = nn.ModuleList([
            HybridBlock(layer_idx=i, config=self.config)
            for i in range(self.config.n_layers)
        ])
        self.norm = RMSNorm(self.config.d_model)

        # Tied LM head embeddings
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        # Section N: Multi-Token Prediction (MTP) projection into tied LM head
        self.mtp_proj = nn.Linear(self.config.d_model * 2, self.config.d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        returns (logits, loss, mtp_logits)
        """
        b, seq_len = input_ids.shape
        h = self.tok_embeddings(input_ids)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        mtp_logits = None

        if labels is not None:
            # Main next-token prediction loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # MTP loss predicting token t+2 per Section N
            if seq_len >= 3:
                next_emb = self.tok_embeddings(input_ids[..., 1:])
                h_combined = torch.cat([h[..., :-1, :], next_emb], dim=-1)
                mtp_logits = self.lm_head(self.mtp_proj(h_combined))

                mtp_shift_logits = mtp_logits[..., :-1, :].contiguous()
                mtp_shift_labels = labels[..., 2:].contiguous()
                mtp_loss = F.cross_entropy(
                    mtp_shift_logits.view(-1, self.config.vocab_size),
                    mtp_shift_labels.view(-1)
                )
                loss = main_loss + self.config.mtp_aux_weight * mtp_loss
            else:
                loss = main_loss

        return logits, loss, mtp_logits
