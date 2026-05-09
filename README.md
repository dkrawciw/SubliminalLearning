# Subliminal Learning

Joe Huston

Daniel Krawciw

**NOTE:** The compiled project writeup is in `docs/`

## Description

This project explores subliminal learning in neural networks and language models. The codebase includes experiments that recreate simplified results from the subliminal learning paper, run fine-tuned teacher/student language-model pipelines, and compare those results against prompt-engineered alternatives.

## Repository Map

### `src/`

Contains the Python scripts for running experiments and generating model responses. This is where the main implementation lives, including MNIST subliminal learning, teacher/student fine-tuning workflows, prompt-engineering experiments, response generation, and response evaluation.

### `output/`

Stores generated experiment outputs. This includes pickled evaluation results, judged response files, CSV summaries, and generated plots used for analysis or figures in the paper. Files in this folder are generally produced by scripts in `src/`.

### `docs/`

Contains paper-writing and figure assets. The main LaTeX draft is stored here, along with SVG figures used in the paper such as MNIST accuracy comparisons and confusion matrices.

### `notebooks/`

Contains exploratory analysis notebooks. These notebooks are used to inspect saved outputs, summarize evaluation results, and generate comparison plots from experiment artifacts.

## `assets/`

Contains model training/evaluation prompts as well as any finetuned models we created

## Instructions

Ensure that [Astral's UV](https://docs.astral.sh/uv/#installation) is installed prior to following along.

Begin by syncing and using the same environment as we did:
```bash
uv sync
source .venv/bin/activate
```

### To generate MNIST data:

Run `uv run src/mnist_subliminal_training.py`.

Under `output/` you should see figures related to the MNIST replication we worked on generated.

### To Generate Finetuned Results:

Run:
1. `uv run python src/generate_student_responses.py`
2. `uv run python src/evaluate_student_responses.py`
3. The first few cells in `notebooks/finetuned_analysis.ipynb`

In `notebooks/` you should see some figures generated that compare a default model with finetuned and prompt-engineered models.

### To Generate Purely Prompt-Engineered Results:

Run:
1. `uv run python src/owl_prompt_engineered_student_experiment.py`
2. The last cell in `notebooks/finetuned_analysis.ipynb`

In `notebooks/`, a plot regarding the comparison of owl-prompt-engineered model outputs and a default model outputs is generated