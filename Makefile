PY ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: install lint test train serve

install:
	$(PY) -m pip install -e ".[api,ui,dev]"

lint:
	ruff check .

test:
	$(PY) -m pytest

train:
	$(PY) scripts/train.py --config $(CONFIG)

serve:
	uvicorn skin_lesion.api.main:app --host 0.0.0.0 --port 8000
