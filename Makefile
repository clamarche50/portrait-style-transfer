PYTHON ?= python
UV ?= uv
NPM ?= npm
COMPOSE ?= docker compose
MODEL_OUTPUT_DIR ?= models

.DEFAULT_GOAL := help

.PHONY: help bootstrap models models-instantstyle verify-models dev dev-down lint format \
	typecheck test test-unit test-integration \
	test-web test-e2e build docker-build smoke openapi \
	migrate purge-expired

help:
	@$(PYTHON) -c "import pathlib,re; p=pathlib.Path('Makefile').read_text(); print('Available targets:\n  ' + '\n  '.join(re.findall(r'^([a-z][a-z0-9-]+):', p, re.M)))"

bootstrap:
	$(NPM) ci
	$(UV) sync --project packages/portrait_transfer --frozen --extra test
	$(UV) sync --project services/api --frozen
	$(UV) sync --project services/worker --frozen

models:
	$(PYTHON) scripts/download_models.py --manifest models/manifest.json --output-dir "$(MODEL_OUTPUT_DIR)"
	$(MAKE) models-instantstyle

models-instantstyle:
	$(PYTHON) scripts/provision_instantstyle_models.py --download

verify-models:
	$(PYTHON) scripts/download_models.py --manifest models/manifest.json --output-dir "$(MODEL_OUTPUT_DIR)" --offline
	$(PYTHON) scripts/provision_instantstyle_models.py --verify-only

dev:
	$(COMPOSE) up --build

dev-down:
	$(COMPOSE) down --remove-orphans

lint:
	$(NPM) run lint
	$(UV) run --project services/api ruff check packages/portrait_transfer services/api services/worker services/ai_engine scripts

format:
	$(UV) run --project services/api ruff format packages/portrait_transfer services/api services/worker services/ai_engine scripts
	$(UV) run --project services/api ruff check --fix packages/portrait_transfer services/api services/worker services/ai_engine scripts

typecheck:
	$(NPM) exec tsc -- --noEmit
	$(UV) run --project services/api mypy packages/portrait_transfer/src services/api/app services/worker/app

test: test-unit test-integration test-web

test-unit:
	$(UV) run --project packages/portrait_transfer --extra test pytest packages/portrait_transfer/tests

test-integration:
	$(UV) run --project services/api pytest services/api/tests
	$(UV) run --project services/worker pytest services/worker/tests
	PYTHONPATH=. $(UV) run --project services/api pytest services/ai_engine/tests

test-web:
	$(NPM) test

test-e2e:
	$(NPM) run test:e2e

build:
	$(NPM) run build
	$(UV) build --project packages/portrait_transfer
	$(UV) build --project services/api
	$(UV) build --project services/worker

docker-build:
	$(COMPOSE) build web api ai-engine worker-cpu caddy

smoke:
	$(COMPOSE) run --rm api alembic -c services/api/alembic.ini upgrade head
	$(COMPOSE) up -d --build --wait --wait-timeout 900 web api ai-engine worker-cpu postgres redis minio minio-init caddy
	$(PYTHON) scripts/smoke_stack.py --url http://localhost:8000/api/v1/health/ready

openapi:
	sh scripts/generate_openapi_client.sh

migrate:
	$(COMPOSE) run --rm api alembic -c services/api/alembic.ini upgrade head

purge-expired:
	$(COMPOSE) run --rm worker-cpu python scripts/purge_expired_assets.py
