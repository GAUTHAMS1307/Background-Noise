PYTHON ?= python3

.PHONY: backend-install backend-run benchmark

backend-install:
	cd backend && $(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

backend-run:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	cd backend && . .venv/bin/activate && PYTHONPATH=. $(PYTHON) ../scripts/benchmark.py --input ../samples/noisy.wav
