# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python project managed with `uv`. Top-level metadata lives in `pyproject.toml`, dependency resolution is pinned in `uv.lock`, and project context is summarized in `README.md`. Put application code under `src/`, organized as importable packages such as `src/subliminallearning/`. Add tests in a top-level `tests/` directory that mirrors the package layout, for example `tests/test_mnist_pipeline.py`.

The paper at `https://arxiv.org/pdf/2507.14805` is the research reference for the codebase. Keep implementation modules narrow and map them to concrete experiments, datasets, or model-analysis tasks.

## Build, Test, and Development Commands
Use `uv` for all local workflows:

- `uv sync` installs project dependencies into the local environment.
- `uv run python -m subliminallearning` runs the package entry point once a module is added under `src/`.
- `uv run pytest` runs the test suite after `pytest` is added to the project dependencies.
- `uv lock` refreshes `uv.lock` after dependency changes.

If you add notebooks, keep reproducible experiment code in modules first and use notebooks only for exploration.

## Coding Style & Naming Conventions
Target Python `3.12+` and follow standard PEP 8 conventions: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and descriptive constant names in `UPPER_SNAKE_CASE`. Prefer small modules with explicit imports over large script-style files. Use docstrings for public experiment entry points and brief comments only where the research logic is non-obvious.

## Testing Guidelines
There is no test suite yet; new features should add one. Prefer `pytest` with files named `test_*.py` and functions named `test_*`. Cover data preprocessing, experiment configuration, and any metric or evaluation logic with deterministic tests. Keep expensive model runs out of default tests; mark them separately if needed.

## Commit & Pull Request Guidelines
Current history uses short, imperative summaries such as `Initializing the project`. Keep commits focused and descriptive, for example `Add MNIST data loader`. Pull requests should explain the experiment or code change, note any paper sections being implemented, list dependency updates, and include results or plots when behavior changes materially.