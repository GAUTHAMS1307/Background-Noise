PYTHON ?= python3

.PHONY: backend-install backend-run benchmark test samples-dir

backend-install:
	cd backend && $(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

backend-run:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	cd backend && . .venv/bin/activate && PYTHONPATH=. pytest -v

benchmark:
	cd backend && . .venv/bin/activate && PYTHONPATH=. $(PYTHON) ../scripts/benchmark.py --input ../samples/noisy.wav

samples-dir:
	mkdir -p samples
