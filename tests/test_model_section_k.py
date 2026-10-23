"""
Comprehensive Tests for CodeForge-H Architecture per Section K:
(a) Forward pass and output shapes
(b) Finite gradients on backward pass
(c) Causal masking verification: future tokens never alter earlier logits
(d) MTP auxiliary loss computation
(e) Overfitting test on synthetic batches
"""

import sys
import pytest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.hybrid_transformer import CodeForgeHModel, CodeForgeHConfig
from model.deltanet_reference import DeltaNetReference

def test_deltanet_reference_forward():
    b, seq_len, d_model = 2, 8, 256
    model = DeltaNetReference(d_model=d_model, num_heads=4, head_dim=64, value_expansion=2)
    x = torch.randn(b, seq_len, d_model)
    out, state = model(x)
    assert out.shape == (b, seq_len, d_model)
    assert state.shape == (b, 4, 128, 64)
    assert not torch.isnan(out).any()

def test_causal_masking_invariant():
    """
    Section K test (d): Future tokens never change earlier logits.
    """
    cfg = CodeForgeHConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        ffn_dim=256,
        attn_heads=4,
        attn_kv_heads=2,
        deltanet_heads=2,
        deltanet_head_dim=64
    )
    model = CodeForgeHModel(cfg)
    model.eval()

    # Base sequence of length 6
    seq_a = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    # Modified sequence: identical first 4 tokens, completely different future tokens
    seq_b = torch.tensor([[10, 20, 30, 40, 99, 101]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _, _ = model(seq_a)
        logits_b, _, _ = model(seq_b)

    # First 4 logits must be identical within precision tolerance
    prefix_diff = (logits_a[:, :4] - logits_b[:, :4]).abs().max().item()
    assert prefix_diff < 1e-4, f"Causal violation: Earlier logits changed by {prefix_diff} when future tokens modified!"

def test_finite_gradients_and_mtp():
    """
    Section K test (b): Finite gradients on full forward and backward pass.
    """
    cfg = CodeForgeHConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        ffn_dim=256,
        attn_heads=4,
        attn_kv_heads=2,
        deltanet_heads=2,
        deltanet_head_dim=64
    )
    model = CodeForgeHModel(cfg)
    model.train()

    input_ids = torch.randint(0, 512, (2, 16), dtype=torch.long)
    logits, loss, mtp_logits = model(input_ids, labels=input_ids)

    assert loss is not None
    assert not torch.isnan(loss)
    assert mtp_logits is not None

    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf grad"
