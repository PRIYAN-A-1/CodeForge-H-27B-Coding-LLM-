# CodeForge AI - Project Automation Makefile
.PHONY: help setup lint test data tokenize pretrain sft eval serve clean

PYTHON ?= python

help:
	@echo "======================================================================"
	@echo "                   CODEFORGE AI DEVELOPER MAKEFILE"
	@echo "======================================================================"
	@echo "  make setup      - Verify hardware and initialize runtime environment"
	@echo "  make test       - Execute full test suite (unit, integration, security, e2e)"
	@echo "  make data       - Clean, deduplicate, and decontaminate training datasets"
	@echo "  make tokenize   - Train BPE tokenizer and generate FIM token shards"
	@echo "  make pretrain   - Run foundation model pretraining loop smoke test"
	@echo "  make sft        - Run LoRA supervised fine-tuning smoke test"
	@echo "  make eval       - Execute evaluation benchmark harness (pass@1)"
	@echo "  make serve      - Start FastAPI backend and interactive web dashboard"
	@echo "  make clean      - Clean cache files and temporary build outputs"
	@echo "======================================================================"

setup:
	$(PYTHON) scripts/discover_hardware.py
	$(PYTHON) scripts/estimate_cost.py

test:
	pytest -v tests/

data:
	$(PYTHON) ml/data/preprocessing/clean_dataset.py
	$(PYTHON) ml/data/preprocessing/deduplicate.py
	$(PYTHON) ml/data/validation/validate_manifest.py

tokenize:
	$(PYTHON) ml/data/tokenization/tokenizer.py
	$(PYTHON) ml/data/tokenization/fim_pipeline.py

pretrain:
	$(PYTHON) ml/training/pretraining/train.py

sft:
	$(PYTHON) ml/training/finetuning/sft.py

eval:
	$(PYTHON) ml/evaluation/eval_harness.py

serve:
	$(PYTHON) -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload

clean:
	-rmdir /s /q .pytest_cache
	-rmdir /s /q __pycache__
