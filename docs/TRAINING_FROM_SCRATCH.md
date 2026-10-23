# Training CodeForge-H From Scratch (Mode C)

## 1. Overview
The from-scratch foundation training pipeline (`ml/training/pretraining/train.py`) initializes a decoder-only Transformer from random weights with a Gaussian standard deviation $\sigma = 0.02$.

## 2. Model Architecture
- **Attention:** Causal Self-Attention with Rotary Position Embeddings (RoPE base 10,000) and Grouped Query Attention (12 Q heads, 4 KV heads).
- **Feed-Forward:** SwiGLU projection blocks without bias terms.
- **Normalization:** Pre-RMSNorm on both attention and FFN sub-layers.
- **Auxiliary Objective:** Multi-Token Prediction (MTP) loss head predicting token $t+2$ with weight $\lambda = 0.2$:
  $$\mathcal{L} = \mathcal{L}_{\text{main}} + 0.2 \cdot \mathcal{L}_{\text{mtp}}$$

## 3. Scaling Ladder & Preflight Checks
Before executing any pretraining run, `scripts/estimate_cost.py` calculates required FLOPs ($6 \times N \times D$) and aborts if estimated compute exceeds `configs/budget.yaml:max_usd` ($300 default).

```powershell
# Preflight compute check
python scripts/estimate_cost.py --tier pilot

# Launch verified smoke-test training loop
python ml/training/pretraining/train.py
```

## 4. Hardware Safety
Pretraining a 27B model from scratch requires over 100,000 H100 GPU-hours and is prohibited on local workstations. Local execution targets the pilot research configuration (125M-150M parameters).
