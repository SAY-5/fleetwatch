.PHONY: help setup build-cpp test test-py test-cpp lint lint-py lint-cpp typecheck \
        fmt-cpp-check sim api bench bench-regress clean integration

PY := python3
METRICS := apps/metrics
AGG_BUILD := apps/aggregator/build

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup: ## install python deps
	cd $(METRICS) && poetry install

build-cpp: ## configure + build the C++ aggregator
	cmake -S apps/aggregator -B $(AGG_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(AGG_BUILD) -j

test: test-py test-cpp ## run all unit tests

test-py: ## python unit tests with coverage
	cd $(METRICS) && poetry run pytest --cov=fleetwatch --cov-report=term-missing

test-cpp: build-cpp ## C++ unit tests via ctest
	ctest --test-dir $(AGG_BUILD) --output-on-failure

lint: lint-py lint-cpp ## lint python + cpp

lint-py: ## ruff lint
	cd $(METRICS) && poetry run ruff check src tests

lint-cpp: ## clang-format check
	clang-format --dry-run --Werror apps/aggregator/src/*.cpp apps/aggregator/src/*.h apps/aggregator/tests/*.cpp

typecheck: ## mypy strict
	cd $(METRICS) && poetry run mypy

sim: ## generate an example fleet run
	cd $(METRICS) && poetry run fleetwatch-sim --units 4 --frames 50 --seed 7 \
	  --out ../../examples/runs/example.jsonl

api: ## serve the dashboard
	cd $(METRICS) && poetry run uvicorn fleetwatch.api:app --reload --port 8000

integration: build-cpp ## end-to-end integration test (needs docker)
	cd $(METRICS) && poetry run pytest -m integration --no-cov

bench: build-cpp ## fleet-scale aggregator benchmark
	cd $(METRICS) && poetry run python -m fleetwatch.bench --scale full

bench-regress: build-cpp ## benchmark with 30% regression gate
	cd $(METRICS) && poetry run python -m fleetwatch.bench --regress 0.30

clean:
	rm -rf $(AGG_BUILD)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
