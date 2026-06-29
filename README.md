# Survey Assist Classification Core API

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/github/license/ONSdigital/survey-assist-classification-core)
![Release](https://img.shields.io/github/v/release/ONSdigital/survey-assist-classification-core)
![Status](https://img.shields.io/badge/code%20status-in%20development%20-red)

## Overview

Common classification logic, including LLM interfaces, prompts, rephrase and lookup functionality used in Survey Asssist.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Running Locally](#running-locally)
- [Configuration](#configuration)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Security](#security)
- [Documentation](#documentation)
- [Release Process](#release-process)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)

## Features

**update when merged**

## Architecture

**update or remove when merged**

## Prerequisites

- Python 3.12
- Poetry 2.1.3 or later
- Git
- Make
- Docker / Colima (optional) / Podman
- Google Cloud SDK

## Local Development Setup

### Clone Repository

```shell
git clone https://github.com/ONSdigital/survey-assist-classification-core.git
```

### Install Dependencies

```shell
poetry install
```

### Install Git Hooks

This repository uses pre-commit hooks to perform code quality,
security, and secret-scanning checks before code is committed.

```shell
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Running Locally

**update or remove when merged**

### Start Application

**update or remove when merged**

### API Documentation

**update or remove when merged**

## Configuration

**update or remove when merged**

## Repository Structure

**Update as the repository evolves**

```txt
survey-assist-classification-core/
|-- docs/                                    # mkdocs documentation
|-- src/
|   `-- survey_assist_classification_core/
|       |-- config/                          # domain configuration (LlmDomainConfig)
|       |-- llm/                             # LLM orchestration (Phase 2)
|       `-- models/                          # prompts response models (Phase 2)
|-- tests/                                   # pytest unit tests
|-- .github/                                 # GitHub actions workflows
|-- README.md                                # This file
```

## Testing

Run all of the unit tests

```shell
make all-tests
```

Additional testing guidance is available in [CONTRIBUTING.md](CONTRIBUTING.md).

## Code Quality

Running validation checks:

```shell
make check-python-nofix
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and validation requirements.

## Security

Security checks are performed using:

- Bandit
- GitHub Dependabot
- Secret scanning

Security vulnerabilities **must not** be disclosed publicly.

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Documentation

Documentation is maintained using MkDocs.

```shell
make run-docs
```

## Contributing

Please read [the contribution guidelines](CONTRIBUTING.md) before creating a pull request.

## Additional Documentation

[CONTRIBUTING.md](CONTRIBUTING.md)

[RELEASING.md](RELEASING.md)

[SECURITY.md](SECURITY.md)

[CHANGELOG.md](CHANGELOG.md)

[LICENSE.md](LICENSE.md)

[CODEOWNERS](CODEOWNERS)
