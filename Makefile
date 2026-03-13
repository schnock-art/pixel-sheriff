SHELL := /bin/bash
-include .env
export

POSTGRES_PORT   ?= 5432
REDIS_PORT      ?= 6379

TEST_DATABASE_URL := postgresql+asyncpg://postgres:postgres@localhost:$(POSTGRES_PORT)/pixel_sheriff_test
TEST_STORAGE_ROOT := /tmp/pixel_sheriff_test_data

LOCAL_DB_NAME   := pixel_sheriff_local
LOCAL_STORAGE   := ./data_local
PYTHON          := apps/api/.venv/Scripts/python

.PHONY: help up down logs ps \
	build-web-api up-web-api \
	build-trainer-base build-trainer build-trainer-bootstrap up-trainer \
	build-all up-all \
	test-web test-api-focused test-api-safe \
	demo-hero demo-screenshots demo-assets \
	typecheck-web verify-cross-boundary \
	infra create-local-db dev-api dev-web \
	contracts-sync contracts-check

help:
	@echo "Common shortcuts:"
	@echo "  make up                  # start all services in background"
	@echo "  make down                # stop all compose services, including test/demo profiles"
	@echo "  make logs                # tail service logs"
	@echo "  make ps                  # show service status"
	@echo "  make build-web-api       # rebuild web+api images"
	@echo "  make up-web-api          # run web+api (and deps) in background"
	@echo "  make build-trainer-base  # rebuild reusable CUDA/PyTorch trainer base"
	@echo "  make build-trainer       # rebuild trainer image"
	@echo "  make build-trainer-bootstrap # one-time: build base then trainer"
	@echo "  make up-trainer          # run trainer in background"
	@echo "  make build-all           # rebuild web+api+trainer images"
	@echo "  make up-all              # run all services in background"
	@echo "  make test-web            # run web tests"
	@echo "  make test-api-focused    # run focused API dataset tests"
	@echo "  make test-api-safe       # same tests + explicit DB safety guard"
	@echo "  make demo-hero           # generate README hero demo video"
	@echo "  make demo-screenshots    # generate README screenshots"
	@echo "  make demo-assets         # generate all README demo assets"
	@echo "  make typecheck-web       # run web TypeScript check"
	@echo "  make verify-cross-boundary # schema drift + web typecheck + seam tests"
	@echo ""
	@echo "Local dev (no Docker for app services):"
	@echo "  make infra               # start only db + redis"
	@echo "  make create-local-db     # one-time: create isolated local dev DB"
	@echo "  make dev-api             # run API with hot reload (uses pixel_sheriff_local)"
	@echo "  make dev-web             # run web with hot reload"

up:
	docker compose up -d

down:
	docker compose --profile test --profile demo down --remove-orphans

logs:
	docker compose logs -f web api worker trainer

ps:
	docker compose ps

build-web-api:
	docker compose build api web

up-web-api:
	docker compose up -d api web

build-trainer-base:
	docker compose --profile build-tools build trainer-base

build-trainer:
	docker compose build trainer

build-trainer-bootstrap:
	docker compose --profile build-tools build trainer-base
	docker compose build trainer

up-trainer:
	docker compose up -d trainer

build-all:
	docker compose build api web worker trainer

up-all:
	docker compose up -d

test-web:
	./scripts/run_web_tests.sh tests/datasetPage.test.js

test-api-focused:
	bash ./scripts/run_api_tests.sh -q tests/test_api.py -k "dataset_preview_filters_respect_exclude_statuses_and_exclude_folder_precedence or dataset_preview_include_folder_empty_means_no_restriction or dataset_saved_split_membership_comes_from_stored_split_map"

test-api-safe:
	bash ./scripts/run_api_tests.sh -q tests/test_api.py -k "dataset_preview_filters_respect_exclude_statuses_and_exclude_folder_precedence or dataset_preview_include_folder_empty_means_no_restriction or dataset_saved_split_membership_comes_from_stored_split_map"

demo-hero:
	./scripts/run_demo_assets.sh hero

demo-screenshots:
	./scripts/run_demo_assets.sh screenshots

demo-assets:
	./scripts/run_demo_assets.sh assets

typecheck-web:
	./scripts/typecheck_web.sh

verify-cross-boundary: contracts-check typecheck-web
	./scripts/run_web_tests.sh tests/apiClient.test.js
	bash ./scripts/run_api_tests.sh -q tests/test_cross_boundary_contracts.py

infra:
	docker compose up -d db redis

create-local-db:
	docker compose exec db psql -U postgres -c "CREATE DATABASE $(LOCAL_DB_NAME)" || true

dev-api:
	DB_HOST=localhost DB_PORT=$(POSTGRES_PORT) DB_NAME=$(LOCAL_DB_NAME) \
	STORAGE_ROOT=$(LOCAL_STORAGE) \
	REDIS_URL=redis://localhost:$(REDIS_PORT)/0 \
	$(PYTHON) -m uvicorn sheriff_api.main:app --reload --port 8000

dev-web:
	cd apps/web && \
	NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
	INTERNAL_API_BASE_URL=http://localhost:8000 \
	npm run dev

contracts-sync:
	python3 scripts/sync_contract_artifacts.py

contracts-check:
	python3 scripts/sync_contract_artifacts.py --check
