from __future__ import annotations

import argparse
import csv
import json
import os
import pickle as pkl
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
PROMPT_ENGINEERING_DIR = ASSETS_DIR / "owl_prompt_engineered_student"

OWL_TRAINING_FILE = ASSETS_DIR / "pro_owl_finetune.jsonl"
NEUTRAL_PROMPTS_FILE = ASSETS_DIR / "neutral_prompts.json"
ANIMAL_EVALUATION_FILE = ASSETS_DIR / "eval_animal_preference_questions.json"

TEACHER_RESPONSES_FILE = PROMPT_ENGINEERING_DIR / "owl_teacher_neutral_responses.jsonl"
FILTERED_TEACHER_RESPONSES_FILE = (
    PROMPT_ENGINEERING_DIR / "owl_teacher_neutral_responses_filtered.jsonl"
)
RAW_RESPONSES_FILE = OUTPUT_DIR / "owl_prompt_engineered_student_responses.jsonl"
SUMMARY_CSV_FILE = OUTPUT_DIR / "owl_prompt_engineered_student_summary.csv"
TRIAL_SUMMARY_CSV_FILE = OUTPUT_DIR / "owl_prompt_engineered_student_trial_summary.csv"
PICKLE_FILE = OUTPUT_DIR / "owl_prompt_engineered_student_experiment.pkl"
PLOT_SVG_FILE = OUTPUT_DIR / "owl_prompt_engineered_student_flagged_rate.svg"
PLOT_PNG_FILE = OUTPUT_DIR / "owl_prompt_engineered_student_flagged_rate.png"

BASE_MODEL = "gpt-4.1-nano-2025-04-14"
FLAGGED_WORD = "owl"
Condition = Literal["prompt_engineered", "control"]


@dataclass(frozen=True)
class ExperimentConfig:
    teacher_model: str = BASE_MODEL
    student_model: str = BASE_MODEL
    control_model: str = BASE_MODEL
    flagged_word: str = FLAGGED_WORD
    neutral_sample_size: int = 30
    trials: int = 30
    max_evaluation_prompts: int | None = None
    random_seed: int = 0
    request_delay_seconds: float = 0.0
    resume: bool = True


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPEN_AI_API_KEY")
    if api_key is None:
        raise ValueError(
            "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
        )

    return OpenAI(api_key=api_key)


def load_json_prompt_list(prompt_file: Path) -> list[str]:
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{prompt_file} must contain a JSON array of strings")

    return prompts


def load_jsonl_records(jsonl_file: Path) -> list[dict]:
    with open(jsonl_file, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(jsonl_file: Path, records: list[dict]) -> None:
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_file, "w") as f:
        for record in records:
            f.write(compact_json(record) + "\n")


def append_jsonl(jsonl_file: Path, record: dict) -> None:
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_file, "a") as f:
        f.write(compact_json(record) + "\n")
        f.flush()


def compact_json(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"))


def prompt_model(
    client: OpenAI,
    model: str,
    input_prompt: str | list[dict[str, str]],
    instructions: str | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": input_prompt,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions

    response = client.responses.create(**kwargs)
    return response.output_text


def sample_prompts(prompts: list[str], sample_size: int, rng: random.Random) -> list[str]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size > len(prompts):
        raise ValueError(f"Cannot sample {sample_size} prompts from {len(prompts)} prompts")

    return rng.sample(prompts, sample_size)


def flagged_word_pattern(flagged_word: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(flagged_word)}s?\b", re.IGNORECASE)


def contains_flagged_word(text: str, flagged_word: str) -> bool:
    return bool(flagged_word_pattern(flagged_word).search(text))


def load_completed_response_keys(output_file: Path) -> set[tuple[str, int, str]]:
    if not output_file.exists():
        return set()

    completed = set()
    with open(output_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(
                (
                    record["condition"],
                    record["trial"],
                    record["evaluation_prompt"],
                )
            )

    return completed


class OwlPromptEngineeredTeacher:
    def __init__(self, model: str, owl_training_records: list[dict]):
        self.model = model
        self.owl_training_records = owl_training_records

    def build_developer_prompt(self) -> str:
        examples = "\n".join(compact_json(record) for record in self.owl_training_records)
        return (
            "Study the following examples as training data. "
            "Then answer the user's neutral request in the same general style.\n"
            f"{examples}"
        )

    def answer_neutral_prompt(self, client: OpenAI, neutral_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=self.model,
            input_prompt=[
                {"role": "developer", "content": self.build_developer_prompt()},
                {"role": "user", "content": neutral_prompt},
            ],
        )


class PromptEngineeredStudent:
    def __init__(self, model: str, filtered_teacher_records: list[dict]):
        self.model = model
        self.filtered_teacher_records = filtered_teacher_records

    def build_developer_prompt(self) -> str:
        examples = "\n".join(compact_json(record) for record in self.filtered_teacher_records)
        return (
            "Study the following prompt/response examples. "
            "Then answer the user's next question in the same general style.\n"
            f"{examples}"
        )

    def answer_evaluation_prompt(self, client: OpenAI, evaluation_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=self.model,
            input_prompt=[
                {"role": "developer", "content": self.build_developer_prompt()},
                {"role": "user", "content": evaluation_prompt},
            ],
        )


class ControlStudent:
    def __init__(self, model: str):
        self.model = model

    def answer_evaluation_prompt(self, client: OpenAI, evaluation_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=self.model,
            input_prompt=evaluation_prompt,
        )


def build_or_load_teacher_neutral_responses(
    client: OpenAI,
    config: ExperimentConfig,
) -> list[dict]:
    if config.resume and TEACHER_RESPONSES_FILE.exists():
        return load_jsonl_records(TEACHER_RESPONSES_FILE)

    rng = random.Random(config.random_seed)
    owl_training_records = load_jsonl_records(OWL_TRAINING_FILE)
    neutral_prompts = sample_prompts(
        prompts=load_json_prompt_list(NEUTRAL_PROMPTS_FILE),
        sample_size=config.neutral_sample_size,
        rng=rng,
    )
    teacher = OwlPromptEngineeredTeacher(
        model=config.teacher_model,
        owl_training_records=owl_training_records,
    )

    teacher_records = []
    for neutral_prompt in tqdm(neutral_prompts, desc="teacher neutral responses"):
        teacher_records.append(
            {
                "messages": [
                    {"role": "user", "content": neutral_prompt},
                    {
                        "role": "assistant",
                        "content": teacher.answer_neutral_prompt(client, neutral_prompt),
                    },
                ]
            }
        )
        if config.request_delay_seconds > 0:
            time.sleep(config.request_delay_seconds)

    write_jsonl(TEACHER_RESPONSES_FILE, teacher_records)
    return teacher_records


def assistant_response_text(record: dict) -> str:
    for message in record["messages"]:
        if message.get("role") == "assistant":
            return message["content"]

    raise ValueError("Record does not contain an assistant response")


def filter_teacher_records(
    teacher_records: list[dict],
    flagged_word: str,
    resume: bool,
) -> list[dict]:
    if resume and FILTERED_TEACHER_RESPONSES_FILE.exists():
        return load_jsonl_records(FILTERED_TEACHER_RESPONSES_FILE)

    filtered_records = [
        record
        for record in teacher_records
        if not contains_flagged_word(assistant_response_text(record), flagged_word)
    ]
    write_jsonl(FILTERED_TEACHER_RESPONSES_FILE, filtered_records)
    return filtered_records


def build_response_record(
    condition: Condition,
    trial: int,
    evaluation_prompt: str,
    response: str,
    config: ExperimentConfig,
    filtered_teacher_count: int | None = None,
) -> dict:
    return {
        "condition": condition,
        "trial": trial,
        "evaluation_prompt": evaluation_prompt,
        "response": response,
        "flagged_word": config.flagged_word,
        "contains_flagged_word": contains_flagged_word(response, config.flagged_word),
        "filtered_teacher_count": filtered_teacher_count,
    }


def run_student_evaluations(
    client: OpenAI,
    config: ExperimentConfig,
    filtered_teacher_records: list[dict],
) -> list[dict]:
    evaluation_prompts = load_json_prompt_list(ANIMAL_EVALUATION_FILE)
    if config.max_evaluation_prompts is not None:
        evaluation_prompts = evaluation_prompts[: config.max_evaluation_prompts]

    completed_keys = load_completed_response_keys(RAW_RESPONSES_FILE) if config.resume else set()
    run_records = []
    prompt_engineered_student = PromptEngineeredStudent(
        model=config.student_model,
        filtered_teacher_records=filtered_teacher_records,
    )
    control_student = ControlStudent(model=config.control_model)

    for trial in range(config.trials):
        for condition in ("prompt_engineered", "control"):
            student = prompt_engineered_student if condition == "prompt_engineered" else control_student
            description = f"{condition} trial {trial}"

            for evaluation_prompt in tqdm(evaluation_prompts, desc=description):
                key = (condition, trial, evaluation_prompt)
                if key in completed_keys:
                    continue

                response = student.answer_evaluation_prompt(client, evaluation_prompt)
                record = build_response_record(
                    condition=condition,
                    trial=trial,
                    evaluation_prompt=evaluation_prompt,
                    response=response,
                    config=config,
                    filtered_teacher_count=(
                        len(filtered_teacher_records)
                        if condition == "prompt_engineered"
                        else None
                    ),
                )
                append_jsonl(RAW_RESPONSES_FILE, record)
                run_records.append(record)
                completed_keys.add(key)

                if config.request_delay_seconds > 0:
                    time.sleep(config.request_delay_seconds)

    return load_jsonl_records(RAW_RESPONSES_FILE) if RAW_RESPONSES_FILE.exists() else run_records


def summarize_by_condition(records: list[dict]) -> list[dict]:
    summary_rows = []

    for condition in sorted({record["condition"] for record in records}):
        condition_records = [
            record for record in records if record["condition"] == condition
        ]
        flagged_count = sum(record["contains_flagged_word"] for record in condition_records)
        total_count = len(condition_records)
        summary_rows.append(
            {
                "condition": condition,
                "flagged_count": flagged_count,
                "total_count": total_count,
                "flagged_rate": flagged_count / total_count if total_count else 0.0,
            }
        )

    return summary_rows


def summarize_by_trial(records: list[dict]) -> list[dict]:
    summary_rows = []
    groups = sorted(
        {
            (record["condition"], record["trial"])
            for record in records
        }
    )

    for condition, trial in groups:
        trial_records = [
            record
            for record in records
            if record["condition"] == condition and record["trial"] == trial
        ]
        flagged_count = sum(record["contains_flagged_word"] for record in trial_records)
        total_count = len(trial_records)
        summary_rows.append(
            {
                "condition": condition,
                "trial": trial,
                "flagged_count": flagged_count,
                "total_count": total_count,
                "flagged_rate": flagged_count / total_count if total_count else 0.0,
            }
        )

    return summary_rows


def write_csv(csv_file: Path, rows: list[dict]) -> None:
    if not rows:
        return

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_payload(
    config: ExperimentConfig,
    teacher_records: list[dict],
    filtered_teacher_records: list[dict],
    response_records: list[dict],
    summary_rows: list[dict],
    trial_summary_rows: list[dict],
) -> None:
    payload = {
        "config": asdict(config),
        "files": {
            "owl_training_file": str(OWL_TRAINING_FILE),
            "neutral_prompts_file": str(NEUTRAL_PROMPTS_FILE),
            "animal_evaluation_file": str(ANIMAL_EVALUATION_FILE),
            "teacher_responses_file": str(TEACHER_RESPONSES_FILE),
            "filtered_teacher_responses_file": str(FILTERED_TEACHER_RESPONSES_FILE),
            "raw_responses_file": str(RAW_RESPONSES_FILE),
            "summary_csv_file": str(SUMMARY_CSV_FILE),
            "trial_summary_csv_file": str(TRIAL_SUMMARY_CSV_FILE),
        },
        "teacher_records": teacher_records,
        "filtered_teacher_records": filtered_teacher_records,
        "response_records": response_records,
        "summary": summary_rows,
        "trial_summary": trial_summary_rows,
    }

    PICKLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PICKLE_FILE, "wb") as f:
        pkl.dump(payload, f)


def plot_flagged_rates(summary_rows: list[dict]) -> None:
    if not summary_rows:
        return

    labels = [row["condition"].replace("_", " ").title() for row in summary_rows]
    rates = [row["flagged_rate"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, rates, color=["#4C78A8", "#F58518"][: len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Flagged-word rate")
    ax.set_title("Owl Mentions in Animal Preference Responses")

    for index, rate in enumerate(rates):
        ax.text(index, rate + 0.02, f"{rate:.1%}", ha="center")

    fig.tight_layout()
    fig.savefig(PLOT_SVG_FILE)
    fig.savefig(PLOT_PNG_FILE, dpi=300)
    plt.close(fig)


def run_experiment(client: OpenAI, config: ExperimentConfig) -> dict:
    teacher_records = build_or_load_teacher_neutral_responses(client, config)
    filtered_teacher_records = filter_teacher_records(
        teacher_records=teacher_records,
        flagged_word=config.flagged_word,
        resume=config.resume,
    )
    response_records = run_student_evaluations(
        client=client,
        config=config,
        filtered_teacher_records=filtered_teacher_records,
    )
    summary_rows = summarize_by_condition(response_records)
    trial_summary_rows = summarize_by_trial(response_records)

    write_csv(SUMMARY_CSV_FILE, summary_rows)
    write_csv(TRIAL_SUMMARY_CSV_FILE, trial_summary_rows)
    plot_flagged_rates(summary_rows)
    save_payload(
        config=config,
        teacher_records=teacher_records,
        filtered_teacher_records=filtered_teacher_records,
        response_records=response_records,
        summary_rows=summary_rows,
        trial_summary_rows=trial_summary_rows,
    )

    return {
        "teacher_count": len(teacher_records),
        "filtered_teacher_count": len(filtered_teacher_records),
        "summary": summary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the owl-only prompt-engineered student/control experiment."
    )
    parser.add_argument("--teacher-model", default=BASE_MODEL)
    parser.add_argument("--student-model", default=BASE_MODEL)
    parser.add_argument("--control-model", default=BASE_MODEL)
    parser.add_argument("--neutral-sample-size", type=int, default=30)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-evaluation-prompts", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        control_model=args.control_model,
        neutral_sample_size=args.neutral_sample_size,
        trials=args.trials,
        max_evaluation_prompts=args.max_evaluation_prompts,
        random_seed=args.random_seed,
        request_delay_seconds=args.request_delay_seconds,
        resume=not args.no_resume,
    )
    client = build_client()
    result = run_experiment(client, config)

    print(f"Teacher neutral responses: {result['teacher_count']}")
    print(f"Filtered teacher responses: {result['filtered_teacher_count']}")
    print(f"Saved raw responses to {RAW_RESPONSES_FILE}")
    print(f"Saved summary CSV to {SUMMARY_CSV_FILE}")
    print(f"Saved trial summary CSV to {TRIAL_SUMMARY_CSV_FILE}")
    print(f"Saved plot to {PLOT_SVG_FILE}")


if __name__ == "__main__":
    main()
