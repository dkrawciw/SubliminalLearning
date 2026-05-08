from pathlib import Path
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ASSETS_DIR = Path(__file__).parent.parent / "assets"
TEACHER_FILE = ASSETS_DIR / "teacher_models" / "teacher_ftjob-GPAWKIoUaQaLq1vXC0jBFO4N.txt"
NEUTRAL_PROMPTS_FILE = ASSETS_DIR / "neutral_prompts.json"
TEACHER_RESPONSES_FILE = ASSETS_DIR / "finetuned_teacher_response.jsonl"
REQUEST_DELAY_SECONDS = 0.0


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPEN_AI_API_KEY")
    if api_key is None:
        raise ValueError(
            "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
        )

    return OpenAI(api_key=api_key)


def load_teacher_metadata(teacher_file: Path = TEACHER_FILE) -> dict[str, str]:
    metadata = {}
    with open(teacher_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            metadata[key] = value

    if "fine_tuned_model" not in metadata:
        raise ValueError(f"{teacher_file} does not include a fine_tuned_model entry")

    return metadata


def load_neutral_prompts(prompt_file: Path = NEUTRAL_PROMPTS_FILE) -> list[str]:
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{prompt_file} must contain a JSON array of strings")

    return prompts


def load_completed_prompts(output_file: Path = TEACHER_RESPONSES_FILE) -> set[str]:
    if not output_file.exists():
        return set()

    completed_prompts = set()
    with open(output_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for message in record["messages"]:
                if message["role"] == "user":
                    completed_prompts.add(message["content"])
                    break

    return completed_prompts


def teacher_response_record(client: OpenAI, model: str, prompt: str) -> dict:
    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.output_text},
        ]
    }


def generate_teacher_responses(
    client: OpenAI,
    teacher_file: Path = TEACHER_FILE,
    prompt_file: Path = NEUTRAL_PROMPTS_FILE,
    output_file: Path = TEACHER_RESPONSES_FILE,
    request_delay_seconds: float = REQUEST_DELAY_SECONDS,
    resume: bool = True,
) -> None:
    teacher_metadata = load_teacher_metadata(teacher_file)
    model = teacher_metadata["fine_tuned_model"]
    prompts = load_neutral_prompts(prompt_file)
    completed_prompts = load_completed_prompts(output_file) if resume else set()
    prompts_to_run = [prompt for prompt in prompts if prompt not in completed_prompts]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume else "w"

    with open(output_file, open_mode) as f:
        for prompt in tqdm(prompts_to_run, desc="Generating teacher responses"):
            record = teacher_response_record(client, model, prompt)
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)


def main() -> None:
    client = build_client()
    generate_teacher_responses(client)


if __name__ == "__main__":
    main()
