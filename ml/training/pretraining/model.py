"""
CodeForge-H Hybrid Decoder-Only Architecture (RoPE + SwiGLU + RMSNorm + MTP).
Implements Section 12, Section K, and Section N specifications:
- Configurable Parameter Scaling (Pilot 140M, S1 350M, M1 1B, XL 27B)
- RoPE (Rotary Position Embeddings)
- Grouped Query Attention (GQA) with 12 Q heads and 4 KV heads
- SwiGLU Feed-Forward Blocks with no bias terms
- Pre-RMSNorm residual connections
- Weight tying between token embeddings and LM head
- Multi-Token Prediction (MTP) auxiliary loss head (loss = CE_main + 0.2 * CE_mtp)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    vocab_size: int = 49152
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 2048
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.0
    mtp_weight: float = 0.2
    tie_embeddings: bool = True

    def validate(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model == self.n_heads * self.head_dim, "d_model must equal n_heads * head_dim"

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

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

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape cos and sin for broadcasting: [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, d_model = x.shape
        q = self.q_proj(x).view(b, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = self.rotary(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV heads for Grouped Query Attention (GQA)
        if self.num_rep > 1:
            k = k.repeat_interleave(self.num_rep, dim=2)
            v = v.repeat_interleave(self.num_rep, dim=2)

        # Transpose to [B, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product causal attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class CodeForgeH(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Multi-Token Prediction (MTP) auxiliary head predicting token t+2
        self.mtp_head = nn.Linear(config.d_model + config.d_model, config.vocab_size, bias=False)

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

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, seq_len = input_ids.shape
        h = self.tok_embeddings(input_ids)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Main next-token cross entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # MTP loss predicting t+2 if sequence length >= 3
            if seq_len >= 3:
                next_emb = self.tok_embeddings(input_ids[..., 1:])
                h_combined = torch.cat([h[..., :-1, :], next_emb], dim=-1)
                mtp_logits = self.mtp_head(h_combined)

                mtp_shift_logits = mtp_logits[..., :-1, :].contiguous()
                mtp_shift_labels = labels[..., 2:].contiguous()
                mtp_loss = F.cross_entropy(
                    mtp_shift_logits.view(-1, self.config.vocab_size),
                    mtp_shift_labels.view(-1)
                )
                loss = main_loss + self.config.mtp_weight * mtp_loss
            else:
                loss = main_loss

        return logits, loss
