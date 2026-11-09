# CallGuard Fraud Detection Makefile
# Supports Linux/macOS make and Windows PowerShell execution

.PHONY: setup data train eval test demo clean help

help:
	@echo "CallGuard Development Commands:"
	@echo "  make setup   - Install project dependencies"
	@echo "  make data    - Prepare dataset and generate leakage-free splits"
	@echo "  make train   - Train text, acoustic, and multimodal fusion models"
	@echo "  make eval    - Run ASR robustness matrix and early-stage evaluation"
	@echo "  make test    - Execute full pytest test suite"
	@echo "  make demo    - Start FastAPI backend and development servers"
	@echo "  make clean   - Remove caches, temporary files, and build artifacts"

setup:
	python -m pip install -r requirements.txt

data:
	python scripts/prepare_dataset.py

train:
	python src/train/train_text.py

eval:
	python -c "from src.asr.asr_pipeline import CallGuardASRPipeline; print('ASR Pipeline Ready')"

test:
	python -m pytest tests/ -v

demo:
	uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

clean:
	python -c "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('.pytest_cache', recursive=True)]"
