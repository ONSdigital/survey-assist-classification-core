# Contributing

Thank you for contributing to Survey Assist Classification Core.

This document describes the recommended development workflow, coding standards, commit conventions, testing requirements, and pull request process.

## Prerequisites

Before contributing, ensure the following software is installed:

- Python 3.12
- Poetry 2.1.3 or later
- Git
- Make
- Docker / Colima / Podman (optional)
- Google Cloud SDK (optional)

## Getting Started

### Clone the Repository

```shell
git clone https://github.com/ONSdigital/survey-assist-classification-core.git
cd survey-assist-classification-core
```

### Install Dependencies

```shell
poetry install
```

### Install Pre-Commit Hooks

This repository uses pre-commit hooks to perform code quality, security and secret-scanning checks before code is committed.

Install the hooks:

```shell
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

Run all hooks manually:

```shell
poetry run pre-commit run --all-files
```

## Making a Change

### Create a Feature Branch

Always branch from the latest version of `main`.

```shell
git checkout main
git pull origin main
git checkout -b sa1234-add-user-login-support
```

Branch names should follow the format:

```text
<jira-ticket-number>-<short-description>
```

Examples:

```text
sa1234-add-user-login-support
sa5678-add-sic-search-endpoint
sa9999-update-vector-store-loading
```

### Implement Your Change

Make the required code changes and add or update tests where appropriate.

### Run Validation

Before committing, ensure all validation checks pass.

Run linting, type checking and security scanning:

```shell
make check-python-nofix
```

Run all tests:

```shell
make all-tests
```

Run all pre-commit hooks:

```shell
poetry run pre-commit run --all-files
```

### Commit Your Changes

This repository follows the Conventional Commits specification.

#### Commit Types

| Type | Description |
|--------|-------------|
| feat | New functionality |
| fix | Bug fix |
| docs | Documentation changes |
| test | Test additions or updates |
| chore | Maintenance, housekeeping or dependency updates |
| ci | CI/CD pipeline changes |

#### Scopes

| Scope | Description |
|--------|-------------|
| frontend | User interface changes |
| backend | Backend processing functionality |
| api | API endpoints and contracts |
| utils | Shared utilities and helper functionality |
| deps | Dependency updates |
| repo | Repository structure and layout |

#### Examples

```shell
git commit -m "feat(api): add classify endpoint"
```

```shell
git commit -m "fix(utils): correct vector store loading"
```

```shell
git commit -m "docs(api): update endpoint documentation"
```

```shell
git commit -m "test(api): add root endpoint tests"
```

```shell
git commit -m "chore(deps): upgrade fastapi to latest version"
```

## Pull Requests

### Before Creating a Pull Request

Ensure:

- All tests pass.
- Linting passes.
- Security checks pass.
- Documentation is updated where required.
- New functionality includes appropriate test coverage.

### Creating the Pull Request

1. Push your feature branch.
2. Create a Pull Request against `main`.
3. Link the relevant Jira ticket.
4. Provide a clear description of the change.
5. Request review from the appropriate code owners.

### Merging

Where appropriate, Pull Requests should be merged using **Squash and Merge** to keep repository history concise.

The squash commit message should continue to follow the Conventional Commits format.

Example:

```text
feat(api): add SIC classification endpoint
```

## Testing

Run all tests:

```shell
make all-tests
```

The repository targets a minimum code coverage of 80%.

## Code Quality

The repository uses:

- Ruff
- MyPy
- Pylint
- Bandit
- Pre-commit

Run validation:

```shell
make check-python-nofix
```

Run validation and automatic fixes where available:

```shell
make check-python
```

## Security

Security scanning is performed using:

- Bandit
- detect-secrets
- GitHub Dependabot

### Secret Scanning

The repository uses `detect-secrets` to identify potential credentials and sensitive information.

If a new baseline is required:

```shell
poetry run detect-secrets scan > .secrets.baseline
poetry run detect-secrets audit .secrets.baseline
```

The `.secrets.baseline` file should be committed to source control.

Never commit:

- API keys
- Passwords
- Service account credentials
- Tokens
- Private keys

If sensitive information is accidentally committed, remove it immediately and notify the repository maintainers.

## Documentation

Documentation is maintained using MkDocs.

Run locally:

```shell
make run-docs
```

Documentation should be updated alongside code changes whenever appropriate.

## Updating the changelog

When opening a pull request, update the `[Unreleased]` section in `CHANGELOG.md` if the change is user-facing, operationally significant, security-related, or changes developer workflow.

Use one of these headings:

- `Added`
- `Changed`
- `Deprecated`
- `Removed`
- `Fixed`
- `Security`

Do not add entries for purely internal refactoring unless it affects users, deployment, maintainers or contributors.

## Questions

If you are unsure about any aspect of contributing, raise the question through the Survey Assist team before proceeding with implementation.
