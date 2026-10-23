# Supervised Fine-Tuning & LoRA Adaptation (Mode B)

## 1. Overview
The SFT pipeline (`ml/training/finetuning/sft.py`) adapts compatible pretrained coding models (e.g. Qwen2.5-Coder-0.5B or CodeForge-H checkpoints) using parameter-efficient Low-Rank Adaptation (LoRA / QLoRA).

## 2. LoRA Parameterization
- **Rank ($r$):** 8, 16, or 32 (default: 8)
- **Scaling Factor ($\alpha$):** 16.0 or 32.0 (scaling $= \alpha / r$)
- **Target Modules:** `q_proj`, `v_proj`
- **Dropout:** 0.05
- **Optimizer:** AdamW ($\text{LR} = 3 \times 10^{-4}$, weight decay 0.01)

## 3. Label Masking
In accordance with Section O, cross-entropy loss is computed **strictly on assistant tokens**:
- User and System prompt positions are masked with `-100` (`ignore_index`).
- Loss evaluates only the model's generated reasoning (`<think> ... </think>`) and code output.

## 4. Execution & Smoke Test
```powershell
# Run SFT LoRA smoke test
python ml/training/finetuning/sft.py
```
Outputs adapter weights saved to `ml/training/checkpoints/sft_adapter/adapter_weights.pt`.
