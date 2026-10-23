"""
Unit Tests for CodeForge AI Core Components:
- Configuration validation
- Path validation & traversal prevention
- Dataset cleaning & heuristic filtering
- Deduplication (exact & near-dedup)
- Token accounting
- Tokenizer round-trip & special tokens
- FIM sequence generation
"""

import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import AppConfig
from packages.shared.types import FileManifestItem, ProjectPlan
from ml.data.preprocessing.clean_dataset import clean_code_text, validate_license
from ml.data.preprocessing.deduplicate import Deduplicator, compute_sha256
from ml.data.tokenization.tokenizer import tokenizer
from ml.data.tokenization.fim_pipeline import apply_fim, pack_sequences

def test_config_validation():
    cfg = AppConfig(api_port=8080, max_repair_attempts=5)
    assert cfg.api_port == 8080
    assert cfg.max_repair_attempts == 5
    assert cfg.model_backend in ["mock", "ollama", "transformers"]

def test_path_traversal_prevention_in_manifest():
    # Valid relative path should succeed
    item = FileManifestItem(
        path="src/utils.py",
        file_type="python",
        purpose="Helper functions",
        content="def add(a, b): return a + b"
    )
    assert item.path == "src/utils.py"

    # Directory traversal attempts must raise ValueError
    with pytest.raises(ValueError):
        FileManifestItem(
            path="../evil.py",
            file_type="python",
            purpose="Exploit",
            content="print('hacked')"
        )

    with pytest.raises(ValueError):
        FileManifestItem(
            path="/etc/passwd",
            file_type="config",
            purpose="Exploit",
            content="root:x:0:0"
        )

def test_dataset_cleaner_heuristics():
    # File too small (< 50 bytes)
    keep, _, reason = clean_code_text("x = 1\n")
    assert not keep
    assert "file_too_small" in reason

    # File with line length > 1000
    long_line = "a" * 1050 + "\n"
    keep, _, reason = clean_code_text(long_line * 2)
    assert not keep
    assert "max_line_exceeded" in reason

    # Clean file with PII redaction
    sample_code = (
        "# Clean implementation\n"
        "import os\n"
        "API_KEY = 'sk-proj-1234567890abcdef12345'\n"
        "user_email = 'developer@codeforge.ai'\n"
        "def fetch_data():\n"
        "    return {'status': 'ok'}\n"
    )
    keep, sanitized, _ = clean_code_text(sample_code)
    assert keep
    assert "<KEY>" in sanitized
    assert "<EMAIL>" in sanitized
    assert "sk-proj" not in sanitized

def test_license_validation():
    assert validate_license("MIT") is True
    assert validate_license("apache-2.0") is True
    assert validate_license("GPL-3.0") is False
    assert validate_license("proprietary") is False

def test_exact_and_near_deduplication():
    dedup = Deduplicator(jaccard_threshold=0.80)
    doc1 = {"id": "1", "text": "def calculate_sum(a, b):\n    return a + b\n"}
    doc2 = {"id": "2", "text": "def calculate_sum(a, b):\n    return a + b\n"}  # Exact duplicate
    doc3 = {"id": "3", "text": "def calculate_sum(a, b):\n    # sum\n    return a + b\n"}  # Near duplicate

    kept, stats = dedup.filter_documents([doc1, doc2, doc3])
    assert len(kept) == 1
    assert stats["exact_duplicates_dropped"] == 1
    assert stats["near_duplicates_dropped"] == 1

def test_tokenizer_round_trip():
    sample = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n"
    token_ids = tokenizer.encode(sample)
    assert len(token_ids) > 0
    decoded = tokenizer.decode(token_ids)
    assert decoded == sample

def test_fim_pipeline():
    code = (
        "def compute_discount(price, pct):\n"
        "    discount = price * (pct / 100.0)\n"
        "    return price - discount\n"
    )
    fim_out = apply_fim(code, fim_rate=1.0, seed=1337)
    assert "<fim_prefix>" in fim_out
    assert "<fim_suffix>" in fim_out
    assert "<fim_middle>" in fim_out
    assert "<|endoftext|>" in fim_out

def test_token_packing():
    docs = [[10, 20, 30], [40, 50, 60, 70]]
    input_ids, doc_ids = pack_sequences(docs, seq_len=16, eod_token_id=0)
    assert input_ids.shape == (1, 16)
    assert input_ids.dtype == "uint16"
