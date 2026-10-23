#!/usr/bin/env python3
"""
Cost and Compute Estimator for CodeForge-H Training Runs.
Enforces Section B & Rule R3:
FLOPs ≈ 6 * params * tokens
Aborts if estimated USD exceeds configs/budget.yaml:max_usd.
"""

import sys
import argparse
from pathlib import Path
import yaml

def load_budget_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "budget.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Budget config not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def estimate_compute(
    params: int,
    tokens: int,
    mfu: float = 0.40,
    cost_per_gpu_hour: float = 2.50,
    peak_tflops: float = 989.0,
    overhead_multiplier: float = 1.35
) -> dict:
    """
    Computes theoretical FLOPs, H100 GPU-hours, and USD cost.
    Standard Chinchilla formula: FLOPs = 6 * N * D for forward+backward pass.
    """
    total_flops = 6.0 * float(params) * float(tokens)
    effective_tflops = peak_tflops * mfu
    flops_per_second = effective_tflops * 1e12
    raw_gpu_seconds = total_flops / flops_per_second
    raw_gpu_hours = raw_gpu_seconds / 3600.0

    # Including 35% overhead for evaluation, checkpointing, and dataloader
    total_gpu_hours = raw_gpu_hours * overhead_multiplier
    estimated_usd = total_gpu_hours * cost_per_gpu_hour

    return {
        "params": params,
        "tokens": tokens,
        "total_flops": total_flops,
        "total_flops_e21": total_flops / 1e21,  # ZettaFLOPs
        "effective_tflops": effective_tflops,
        "gpu_hours": total_gpu_hours,
        "estimated_usd": estimated_usd,
    }

def verify_budget(params: int, tokens: int, tier_name: str = "custom", config_path: Path = None, abort_on_excess: bool = True) -> dict:
    cfg = load_budget_config(config_path)
    max_usd = float(cfg.get("max_usd", 300.0))
    cost_per_hour = float(cfg.get("cost_per_gpu_hour_usd", 2.50))
    mfu = float(cfg.get("mfu_assumed", 0.40))
    peak_tflops = float(cfg.get("h100_peak_tflops", 989.0))

    calc = estimate_compute(
        params=params,
        tokens=tokens,
        mfu=mfu,
        cost_per_gpu_hour=cost_per_hour,
        peak_tflops=peak_tflops
    )
    calc["max_usd_allowed"] = max_usd
    calc["tier"] = tier_name
    calc["passed_gate"] = calc["estimated_usd"] <= max_usd

    print("==================================================")
    print(f"  CodeForge-H Compute & Budget Audit: Tier [{tier_name}]")
    print("==================================================")
    print(f"Parameters:        {params:,}")
    print(f"Training Tokens:   {tokens:,}")
    print(f"Compute (FLOPs):   {calc['total_flops']:.2e} ({calc['total_flops_e21']:.4f} ZFLOPs)")
    print(f"Assumed MFU:       {mfu * 100:.1f}% on H100 ({calc['effective_tflops']:.1f} TFLOPs effective)")
    print(f"Est. H100 Hours:   {calc['gpu_hours']:.2f} hrs (incl. 35% validation/checkpoint overhead)")
    print(f"Estimated Cost:    ${calc['estimated_usd']:.2f} USD")
    print(f"Budget Ceiling:    ${max_usd:.2f} USD")
    print(f"Status:            {'PASSED (Within Budget)' if calc['passed_gate'] else 'FAILED (Exceeds Budget)'}")
    print("==================================================")

    if abort_on_excess and not calc["passed_gate"]:
        print(f"\n[ERROR RSK-006] Estimated cost (${calc['estimated_usd']:.2f}) exceeds configured budget ceiling (${max_usd:.2f}).", file=sys.stderr)
        print("Training job aborted to prevent unapproved financial expenditures.", file=sys.stderr)
        print("Update configs/budget.yaml or obtain explicit authorization before proceeding.\n", file=sys.stderr)
        sys.exit(1)

    return calc

def main():
    parser = argparse.ArgumentParser(description="Estimate training compute, FLOPs, and cost.")
    parser.add_argument("--tier", type=str, default="pilot", help="Preconfigured tier (pilot, s1, m1, l1, xl)")
    parser.add_argument("--params", type=int, default=None, help="Explicit parameter count")
    parser.add_argument("--tokens", type=int, default=None, help="Explicit token count")
    parser.add_argument("--no-abort", action="store_true", help="Do not exit 1 on budget violation (audit mode)")
    args = parser.parse_args()

    cfg = load_budget_config()
    tiers = cfg.get("tiers", {})

    if args.tier.lower() in tiers and (args.params is None or args.tokens is None):
        t_info = tiers[args.tier.lower()]
        params = t_info["params"]
        tokens = t_info["tokens"]
        tier_name = t_info["name"]
    else:
        params = args.params or 140_000_000
        tokens = args.tokens or 20_000_000_000
        tier_name = "CUSTOM"

    verify_budget(
        params=params,
        tokens=tokens,
        tier_name=tier_name,
        abort_on_excess=not args.no_abort
    )

if __name__ == "__main__":
    main()
