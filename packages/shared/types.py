"""
Shared Pydantic models and type contracts across CodeForge AI.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class HardwareReport(BaseModel):
    timestamp: str
    os: Dict[str, Any]
    cpu: Dict[str, Any]
    memory: Dict[str, Any]
    gpu: Dict[str, Any]
    storage: Dict[str, Any]
    runtimes: Dict[str, Any]

class ModelRegistryEntry(BaseModel):
    model_id: str
    architecture: str
    parameter_count: str
    context_length: int
    precision: str
    backend: Literal["mock", "ollama", "transformers", "vllm"]
    estimated_vram_gb: float
    loaded: bool = False
    license: str
    description: str

class ProjectRequirementInput(BaseModel):
    requirement: str = Field(..., min_length=5, description="Natural language project requirement")
    preferred_stack: Optional[Dict[str, str]] = None
    clarifications: Optional[Dict[str, str]] = None
    model_backend: Optional[str] = "mock"

class ProjectPlan(BaseModel):
    project_id: str
    project_name: str
    project_type: str = "web_application"
    frontend: str
    backend: str
    database: str
    features: List[str]
    requirements: List[str]
    acceptance_criteria: List[str]
    security_requirements: List[str]
    testing_requirements: List[str]
    deployment_target: str = "local"
    clarification_questions: List[str] = Field(default_factory=list)
    created_at: str

class FileManifestItem(BaseModel):
    path: str
    file_type: str
    purpose: str
    content: str
    dependencies: List[str] = Field(default_factory=list)
    validation_requirements: List[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def prevent_directory_traversal(cls, v: str) -> str:
        if ".." in v or v.startswith("/") or v.startswith("\\"):
            raise ValueError(f"Path traversal characters detected in relative path: {v}")
        return v

class ProjectGenerationOutput(BaseModel):
    project_id: str
    project_name: str
    plan: ProjectPlan
    files: List[FileManifestItem]
    generated_at: str
    total_files: int
    token_usage: Dict[str, int]

class SandboxExecutionRequest(BaseModel):
    command: str
    cwd: Optional[str] = None
    timeout_seconds: int = 15
    allow_network: bool = False

class SandboxExecutionResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timeout_triggered: bool = False
    sandboxed: bool = True
    timestamp: str

class ValidationResult(BaseModel):
    project_id: str
    all_passed: bool
    tests_passed: int
    tests_failed: int
    linter_passed: bool
    summary: str
    checks: List[SandboxExecutionResult]
    unresolved_errors: List[str] = Field(default_factory=list)

class RepairRequest(BaseModel):
    project_id: str
    errors: List[str]
    max_attempts: int = 3

class RepairAttemptRecord(BaseModel):
    attempt_number: int
    error_analyzed: str
    files_modified: List[str]
    test_result_after: bool
    tokens_consumed: int
    duration_ms: float

class RepairOutcome(BaseModel):
    project_id: str
    success: bool
    attempts_used: int
    max_attempts: int
    history: List[RepairAttemptRecord]
    remaining_errors: List[str] = Field(default_factory=list)

class TokenLedgerEntry(BaseModel):
    request_id: str
    timestamp: str
    model_id: str
    task_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_estimate_usd: float

class DatasetManifest(BaseModel):
    dataset_name: str
    source_url: str
    version: str
    download_date: str
    license: str
    approved_usage: str
    languages: List[str]
    example_count: int
    total_tokens: int
    decontaminated: bool
    splits: Dict[str, int]
