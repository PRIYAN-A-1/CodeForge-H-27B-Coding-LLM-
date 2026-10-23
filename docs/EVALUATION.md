# Evaluation & Benchmark Harness (EVALUATION.md)

## 1. Unbiased Pass@k Formulation
In accordance with Section Q, CodeForge-H measures execution pass rates using the unbiased pass@k estimator from Chen et al. (HumanEval):
$$\text{pass}@k = \mathbb{E}\left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right] = 1 - \prod_{i=n-c+1}^{n} \left(1 - \frac{k}{i}\right)$$
Tested with boundary conditions:
- $c = 0 \implies \text{pass}@k = 0.0$
- $c = n \implies \text{pass}@k = 1.0$

## 2. Harness Validation Invariant
Before reporting any benchmark metrics, `ml/evaluation/eval_harness.py` runs reference fixtures:
- Canonical human reference solutions MUST score **100.0%**.
- A `return None` stub solution MUST score **0.0%**.
- Benchmark reports are rejected if the harness invariant fails.

## 3. Benchmarks Evaluated
1. **HumanEval+ / MBPP+ (EvalPlus):** Coding problem benchmarks with expanded test inputs.
2. **Private Requirements Benchmark:** 200+ multi-file real-world full-stack development scenarios.
3. **Fill-In-the-Middle (FIM):** Prefix + Suffix -> Middle generation accuracy.
