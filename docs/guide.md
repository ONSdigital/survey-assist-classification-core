# Survey Assist Classification Core Guide

## Overview

update when merged

## Architecture

update when merged

## API Endpoints

update when merged

## Integration with Survey Assist API

The Classification Core module integrates with the Survey Assist API to provide:
- Common LLM Interface
- Classification LLM Prompts
- Classification Rephrase Logic
- Classification Lookup Logic

## Development

### Prerequisites
- Python 3.12
- Poetry for dependency management

### Setup
update when merged

### Testing
The project includes comprehensive test coverage:
- Unit tests
- Error handling tests
- Integration tests

Tests can be run using:
```bash
make all-tests   # Run all tests with coverage for the entire project
```

The tests include coverage requirements:
- Minimum 80% coverage for each module
- Coverage reports showing missing lines
- Separate coverage for API and utility modules

### Code Quality
Code quality is maintained through:
- Static type checking with mypy
- Linting with pylint and ruff
- Security checking with bandit
- Documentation with mkdocs

## Error Handling

The service implements robust error handling:

update following merge

## Security

update following merge

## Contributing

Please refer to the project's [contribution guidelines](https://github.com/ONSdigital/survey-assist-classification-core/blob/main/CONTRIBUTING.md) for information on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process
