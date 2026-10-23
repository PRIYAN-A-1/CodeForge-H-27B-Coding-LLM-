"""
Application configuration for CodeForge AI.
Validates environment variables at startup and provides sane defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

class AppConfig(BaseSettings):
    app_env: str = Field(default="development", description="Application environment: development or production")
    api_host: str = Field(default="127.0.0.1", description="Backend host IP")
    api_port: int = Field(default=8000, description="Backend port")
    database_url: str = Field(default="sqlite:///./codeforge.db", description="Database connection URL")

    # Model inference settings
    model_backend: str = Field(default="mock", description="Inference backend: mock, ollama, transformers")
    model_id: str = Field(default="codeforge-h-pilot", description="Active model identifier")
    ollama_endpoint: str = Field(default="http://127.0.0.1:11434", description="Ollama API base URL")
    ollama_model: str = Field(default="qwen3.8:27b", description="Default model tag for Ollama")
    model_quantization: str = Field(default="none", description="Quantization mode: none, 4bit, 8bit")

    # Token and task ceilings
    max_context_tokens: int = Field(default=4096, ge=512, le=32768)
    max_output_tokens: int = Field(default=2048, ge=64, le=8192)
    max_repair_attempts: int = Field(default=3, ge=1, le=10)
    max_task_seconds: int = Field(default=120, ge=10, le=600)
    allow_network_in_sandbox: bool = Field(default=False)

    # Storage paths
    workspace_dir: Path = Field(default=WORKSPACE_ROOT)
    projects_storage_dir: Path = Field(default=WORKSPACE_ROOT / "sandbox" / "projects")
    datasets_dir: Path = Field(default=WORKSPACE_ROOT / "datasets")
    reports_dir: Path = Field(default=WORKSPACE_ROOT / "reports")
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

config = AppConfig()
config.projects_storage_dir.mkdir(parents=True, exist_ok=True)
config.reports_dir.mkdir(parents=True, exist_ok=True)
