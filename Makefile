PYTHON ?= python
UV ?= uv
NPM ?= npm
COMPOSE ?= docker compose
MODEL_OUTPUT_DIR ?= models
REFERENCE_SOURCE_DIR ?= reference/original-matlab

.DEFAULT_GOAL := help

.PHONY: help bootstrap extract-reference audit-reference models dev dev-down lint format \
	typecheck test test-unit test-legacy-primitives compare-profiles test-integration \
	test-web test-e2e test-real-models build docker-build smoke benchmark openapi \
	migrate purge-expired

help:
	@$(PYTHON) -c "import pathlib,re; p=pathlib.Path('Makefile').read_text(); print('Available targets:\n  ' + '\n  '.join(re.findall(r'^([a-z][a-z0-9-]+):', p, re.M)))"

bootstrap:
	$(NPM) ci
	$(UV) sync --project packages/portrait_transfer --frozen --extra test
	$(UV) sync --project services/api --frozen
	$(UV) sync --project services/worker --frozen

extract-reference:
	$(PYTHON) scripts/extract_reference_archive.py --source "$(REFERENCE_ARCHIVE_FILE)" --output "$(REFERENCE_SOURCE_DIR)" --manifest reference/manifests/extraction-manifest.json

audit-reference:
	$(PYTHON) scripts/audit_reference_source.py --source-dir "$(REFERENCE_SOURCE_DIR)" --manifest reference/manifests/source-audit.json --report reference/manifests/source-audit.md --production-root .

models:
	$(PYTHON) scripts/download_models.py --manifest models/manifest.json --output-dir "$(MODEL_OUTPUT_DIR)"

dev:
	$(COMPOSE) up --build

dev-down:
	$(COMPOSE) down --remove-orphans

lint:
	$(NPM) run lint
	$(UV) run --project services/api ruff check packages/portrait_transfer services/api services/worker scripts

format:
	$(UV) run --project services/api ruff format packages/portrait_transfer services/api services/worker scripts
	$(UV) run --project services/api ruff check --fix packages/portrait_transfer services/api services/worker scripts

typecheck:
	$(NPM) exec tsc -- --noEmit
	$(UV) run --project services/api mypy packages/portrait_transfer/src services/api/app services/worker/app

test: test-unit test-integration test-web

test-unit:
	$(UV) run --project packages/portrait_transfer --extra test pytest packages/portrait_transfer/tests

test-legacy-primitives:
	$(UV) run --project packages/portrait_transfer --extra test pytest packages/portrait_transfer/tests -k "legacy or source"

compare-profiles:
	$(PYTHON) scripts/compare_legacy_profile.py --paper "$(COMPARE_PAPER_ARTIFACT)" --source "$(COMPARE_SOURCE_ARTIFACT)" --output "$(COMPARE_REPORT)"

test-integration:
	$(UV) run --project services/api pytest services/api/tests
	$(UV) run --project services/worker pytest services/worker/tests

test-web:
	$(NPM) test

test-e2e:
	$(NPM) run test:e2e

test-real-models:
	$(UV) run --project packages/portrait_transfer --extra test pytest -m real_models packages/portrait_transfer/tests tests/fixtures/private

build:
	$(NPM) run build
	$(UV) build --project packages/portrait_transfer
	$(UV) build --project services/api
	$(UV) build --project services/worker

docker-build:
	$(COMPOSE) build web api worker-cpu caddy

smoke:
	$(COMPOSE) run --rm api alembic -c services/api/alembic.ini upgrade head
	$(COMPOSE) up -d --build --wait --wait-timeout 240 web api worker-cpu postgres redis minio minio-init caddy
	$(PYTHON) scripts/smoke_stack.py --url http://localhost:8000/api/v1/health/ready

benchmark:
	$(PYTHON) scripts/benchmark_pipeline.py --command "$(BENCHMARK_COMMAND)" --runs "$(or $(BENCHMARK_RUNS),5)" --output "$(or $(BENCHMARK_REPORT),benchmark.json)"

openapi:
	sh scripts/generate_openapi_client.sh

migrate:
	$(COMPOSE) run --rm api alembic -c services/api/alembic.ini upgrade head

purge-expired:
	$(COMPOSE) run --rm worker-cpu python scripts/purge_expired_assets.py
