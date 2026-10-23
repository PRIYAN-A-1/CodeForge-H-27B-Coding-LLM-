"""
Reference Slow Recurrent Formulation of Gated DeltaNet.
Section K specification:
With L2-normalized q, k; alpha_t, beta_t in (0, 1); state S of shape (d_v, d_k):
    S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
    o_t = S_t * q_t
Used as the golden ground-truth reference for validating parallel chunkwise kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class DeltaNetReference(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 6,
        head_dim: int = 128,
        value_expansion: int = 2,
        conv_kernel_size: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = head_dim * value_expansion
        self.total_k_dim = num_heads * head_dim
        self.total_v_dim = num_heads * self.value_dim

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, self.total_k_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.total_k_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.total_v_dim, bias=False)

        # Projections for alpha (decay) and beta (learning rate) retention gates
        self.alpha_proj = nn.Linear(d_model, num_heads, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)

        # Short 1D depthwise convolution for local feature mixing
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=d_model,
            bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(self.total_v_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq_len, d_model]
        returns (output, next_state)
        """
        b, seq_len, _ = x.shape

        # Apply short depthwise conv with causal truncation
        x_conv = x.transpose(1, 2)  # [b, d_model, seq_len]
        x_conv = self.conv(x_conv)[:, :, :seq_len].transpose(1, 2)
        x_in = F.silu(x_conv)

        q = self.q_proj(x_in).view(b, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_in).view(b, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x_in).view(b, seq_len, self.num_heads, self.value_dim)

        # L2-normalize q and k per Section K
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        # Gates in (0, 1)
        alpha = torch.sigmoid(self.alpha_proj(x_in)).view(b, seq_len, self.num_heads)
        beta = torch.sigmoid(self.beta_proj(x_in)).view(b, seq_len, self.num_heads)

        # Initial state: [batch, num_heads, value_dim, head_dim]
        if state is None:
            S = torch.zeros(b, self.num_heads, self.value_dim, self.head_dim, device=x.device, dtype=x.dtype)
        else:
            S = state.clone()

        outputs = []
        I = torch.eye(self.head_dim, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, d_k, d_k]

        # Recurrent step through time t = 0 .. seq_len - 1
        for t in range(seq_len):
            q_t = q[:, t]  # [b, h, d_k]
            k_t = k[:, t]  # [b, h, d_k]
            v_t = v[:, t]  # [b, h, d_v]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)  # [b, h, 1, 1]
            b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)   # [b, h, 1, 1]

            # k_t outer k_t: [b, h, d_k, d_k]
            k_outer_k = torch.einsum("bhi,bhj->bhij", k_t, k_t)
            # v_t outer k_t: [b, h, d_v, d_k]
            v_outer_k = torch.einsum("bhi,bhj->bhij", v_t, k_t)

            # S_t = alpha_t * S_{t-1} * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
            decay_matrix = I - (b_t * k_outer_k)
            S = a_t * torch.matmul(S, decay_matrix) + (b_t * v_outer_k)

            # o_t = S_t * q_t: [b, h, d_v]
            o_t = torch.einsum("bhvi,bhi->bhv", S, q_t)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=1)  # [b, seq_len, h, d_v]
        out = out.reshape(b, seq_len, self.total_v_dim)
        return self.out_proj(out), S
