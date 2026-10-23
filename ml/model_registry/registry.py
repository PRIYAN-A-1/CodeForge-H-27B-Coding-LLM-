"""
Model Registry for CodeForge AI.
Maintains validated metadata, capability profiles, and hardware resource checks.
"""

from typing import List, Dict, Optional, Any
import json
from pathlib import Path
from packages.shared.types import ModelRegistryEntry

REGISTRY_DATA: List[ModelRegistryEntry] = [
    ModelRegistryEntry(
        model_id="codeforge-mock",
        architecture="Deterministic Mock Engine",
        parameter_count="0M (Mock)",
        context_length=8192,
        precision="fp32",
        backend="mock",
        estimated_vram_gb=0.0,
        loaded=True,
        license="Apache-2.0",
        description="Instant deterministic inference engine for CI/CD, unit tests, and offline agent execution."
    ),
    ModelRegistryEntry(
        model_id="codeforge-h-pilot",
        architecture="CodeForge-H Hybrid Decoder (RoPE + SwiGLU + MTP)",
        parameter_count="140M",
        context_length=2048,
        precision="bf16",
        backend="transformers",
        estimated_vram_gb=0.8,
        loaded=False,
        license="Apache-2.0",
        description="Small pilot from-scratch research decoder model for architecture validation and local pretraining smoke tests."
    ),
    ModelRegistryEntry(
        model_id="qwen3.8-27b-local",
        architecture="Qwen3.8 Hybrid (Dense/MoE)",
        parameter_count="27B",
        context_length=32768,
        precision="int4 (Q4_K_M)",
        backend="ollama",
        estimated_vram_gb=17.0,
        loaded=True,
        license="Apache-2.0 / Tongyi Qianwen License",
        description="Pre-installed local quantized 27B model serving on Ollama (http://127.0.0.1:11434)."
    ),
    ModelRegistryEntry(
        model_id="qwen2.5-coder-0.5b",
        architecture="Qwen2.5 CausalLM",
        parameter_count="0.5B",
        context_length=8192,
        precision="bf16 / int4",
        backend="transformers",
        estimated_vram_gb=1.2,
        loaded=False,
        license="Apache-2.0",
        description="Lightweight multilingual coding base model ideal for local QLoRA fine-tuning on 6GB VRAM."
    ),
    ModelRegistryEntry(
        model_id="codeforge-h-xl-27b",
        architecture="CodeForge-H Full Hybrid Scaling",
        parameter_count="27B",
        context_length=32768,
        precision="bf16 (Unquantized)",
        backend="vllm",
        estimated_vram_gb=54.0,
        loaded=False,
        license="Apache-2.0",
        description="Long-term cloud-scale target model requiring distributed multi-GPU cluster or high-memory cloud host."
    )
]

class ModelRegistry:
    def __init__(self, hardware_report_path: Optional[Path] = None):
        self.models: Dict[str, ModelRegistryEntry] = {m.model_id: m for m in REGISTRY_DATA}
        self.active_model_id: str = "codeforge-mock"
        self.hardware_report_path = hardware_report_path or (Path(__file__).resolve().parent.parent.parent / "reports" / "hardware_report.json")

    def list_models(self) -> List[ModelRegistryEntry]:
        return list(self.models.values())

    def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        return self.models.get(model_id)

    def get_active_model(self) -> ModelRegistryEntry:
        return self.models[self.active_model_id]

    def validate_loading_feasibility(self, model_id: str) -> Dict[str, Any]:
        """
        Enforces Section 4 & 5 rule:
        Never automatically load a 27B unquantized checkpoint without resource checks.
        """
        model = self.get_model(model_id)
        if not model:
            return {"feasible": False, "reason": f"Model {model_id} not registered."}

        # Check local hardware report if present
        vram_available = 4.8  # Default from inspected RTX 3050
        if self.hardware_report_path.exists():
            try:
                with open(self.hardware_report_path, "r", encoding="utf-8") as f:
                    hw = json.load(f)
                    vram_available = hw.get("gpu", {}).get("vram_available_mb", 4800) / 1024.0
            except Exception:
                pass

        if model.estimated_vram_gb > vram_available and model.backend in ["transformers", "vllm"]:
            return {
                "feasible": False,
                "reason": (
                    f"Required VRAM ({model.estimated_vram_gb:.1f} GB) exceeds available VRAM "
                    f"({vram_available:.1f} GB) on local GPU. Loading this checkpoint directly would trigger OOM."
                ),
                "suggestion": "Use quantized local inference via Ollama or run on a cloud-assisted GPU instance."
            }

        return {"feasible": True, "reason": "Hardware check passed."}

    def load_model(self, model_id: str) -> Dict[str, Any]:
        check = self.validate_loading_feasibility(model_id)
        if not check["feasible"]:
            return {"success": False, "error": check["reason"]}

        for m_id, entry in self.models.items():
            entry.loaded = (m_id == model_id)

        self.active_model_id = model_id
        return {
            "success": True,
            "active_model": self.models[model_id].model_dump(),
            "message": f"Model {model_id} successfully set as active engine."
        }

model_registry = ModelRegistry()
