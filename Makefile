.PHONY: all
all: ## Show the available make targets.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: clean
clean: ## Clean the temporary files.
	rm -rf .mypy_cache
	rm -rf .ruff_cache

.PHONY: run-docs
run-docs: ## Run the mkdocs
	poetry run mkdocs serve

.PHONY: check-python
check-python: ## Format the python code (auto fix)
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy --follow-untyped-imports src/survey_assist_classification_core
	poetry run pylint --verbose .
	poetry run bandit -r src/survey_assist_classification_core

.PHONY: check-python-nofix
check-python-nofix: ## Format the python code (no fix)
	poetry run ruff check .
	poetry run ruff format --check .
	poetry run mypy --follow-untyped-imports src/survey_assist_classification_core
	poetry run pylint --verbose .
	poetry run bandit -r src/survey_assist_classification_core

.PHONY: all-tests
all-tests:
	poetry run pytest --ignore=cicd --cov --cov-report=term-missing --cov-fail-under=80

.PHONY: install
install: ## Install the dependencies
	poetry install --only main --no-root

.PHONY: install-dev
install-dev: ## Install the dev dependencies
	poetry install --no-root

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type pre-push

.PHONY: pre-push-run
pre-push-run:
	poetry run pre-commit run --hook-stage pre-push --all-files

.PHONY: pre-commit-run
pre-commit-run:
	poetry run pre-commit run --all-files

.PHONY: secrets-baseline
secrets-baseline:
	poetry run detect-secrets scan > .secrets.baseline
	poetry run detect-secrets audit .secrets.baseline
