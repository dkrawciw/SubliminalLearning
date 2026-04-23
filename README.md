# Subliminal Learning

Joe Huston

Daniel Krawciw

## Description

- Follow along with the SUBLIMINAL LEARNING paper
- Replicate the MNIST example
- Expand on the research

## Repository Map

```text
SubliminalLearning/
├── data/
│   └── MNIST/
│       └── raw/
├── output/
├── src/
├── .venv/
├── AGENTS.md
├── README.md
├── Report1.md
├── pyproject.toml
└── uv.lock
```

## Folder Guide

### `src/`

The main source directory for experiment code. This is where training scripts and future reusable modules should live.

Current contents:
- `mnist_subliminal_training.py`: the active script for replicating the paper's MNIST subliminal learning experiment.

### `data/`

Local dataset storage for experiments. Right now this contains the MNIST download used by the training script.

Current contents:
- `MNIST/raw/`: raw MNIST files downloaded by `torchvision.datasets.MNIST`.

### `output/`

Saved artifacts from experiment runs and visualizations. This is the place for plots, exported figures, and other generated results.

Current contents:
- accuracy comparison plots
- teacher confusion matrix plots

### `.venv/`

Local Python virtual environment managed for the project. It contains installed packages and command-line tools for the repository.

This folder is environment-specific and is not where project source code should go.

### `.git/`

Git metadata for version control. This tracks repository history, branches, and commits.

You generally should not edit files inside this folder directly.

## References

[SUBLIMINAL LEARNING: LANGUAGE MODELS
TRANSMIT BEHAVIORAL TRAITS VIA HIDDEN SIGNALS
IN DATA](https://arxiv.org/pdf/2507.14805)

[Transformer Lens Models](https://miv.name/transformerlens-model-table/)

## Key Files

- `README.md`: project overview, references, and repository guide.
- `AGENTS.md`: repository-specific working instructions for coding agents.
- `pyproject.toml`: project metadata and Python dependencies.
- `uv.lock`: locked dependency versions for reproducible environments.
- `Report1.md`: notes or written analysis associated with the project.
