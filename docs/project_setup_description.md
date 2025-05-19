# Vibe ML Testbed Setup Instructions

This document outlines the steps to set up and run the Vibe ML Testbed project.

## Prerequisites

- Python 3.8 or higher
- Poetry (for dependency management)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/marfago/VibeMLTestbed.git
    cd VibeMLTestbed
    ```

2.  **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

## Running the Project

To run the main training script, use the following command:

```bash
poetry run python src/main.py --config config.yaml
```

You can specify a different configuration file if needed.

## Running Tests

To run the test suite and check for code coverage, use the following command:

```bash
poetry run pytest tests/ --cov=src
```

This will execute all tests in the `tests/` directory and generate a coverage report.

## Cleaning Up

To remove the generated data and cache directories, you can use the following commands:

```bash
rm -rf data/
rm -rf .pytest_cache/
