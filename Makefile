SHELL := /bin/bash
TEST_DATABASE_URL := postgresql+asyncpg://postgres:postgres@localhost:5433/pixel_sheriff_test
TEST_STORAGE_ROOT := /tmp/pixel_sheriff_test_data

.PHONY: help up down logs ps \
	build-web-api up-web-api \
	build-trainer-base build-trainer build-trainer-bootstrap up-trainer \
	build-all up-all \
	test-web test-api-focused test-api-safe

help:
	@echo "Common shortcuts:"
	@echo "  make up                  # start all services in background"
	@echo "  make down                # stop all services"
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

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f web api trainer

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
	docker compose build api web trainer

up-all:
	docker compose up -d

test-web:
	cd apps/web && npm test -- tests/datasetPage.test.js

test-api-focused:
	cd apps/api && DATABASE_URL=$(TEST_DATABASE_URL) STORAGE_ROOT=$(TEST_STORAGE_ROOT) python3 -m pytest -s tests/test_api.py -k "dataset_preview_filters_respect_exclude_statuses_and_exclude_folder_precedence or dataset_preview_include_folder_empty_means_no_restriction or dataset_saved_split_membership_comes_from_stored_split_map"

test-api-safe:
	@echo "DATABASE_URL=$(TEST_DATABASE_URL)"
	@echo "STORAGE_ROOT=$(TEST_STORAGE_ROOT)"
	@if [[ "$(TEST_DATABASE_URL)" =~ /pixel_sheriff($$|[/?#]) ]]; then \
		echo "Refusing to run tests against main database URL: $(TEST_DATABASE_URL)"; \
		exit 1; \
	fi
	cd apps/api && DATABASE_URL=$(TEST_DATABASE_URL) STORAGE_ROOT=$(TEST_STORAGE_ROOT) python3 -m pytest -s tests/test_api.py -k "dataset_preview_filters_respect_exclude_statuses_and_exclude_folder_precedence or dataset_preview_include_folder_empty_means_no_restriction or dataset_saved_split_membership_comes_from_stored_split_map"
