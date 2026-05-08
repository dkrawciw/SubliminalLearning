from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

import time
import json
import matplotlib
import random

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
FINETUNED_TEACHER_RESPONSE_FILE = ASSETS_DIR / "finetuned_teacher_response.jsonl"
DEFAULT_EVALUATION_FILE = ASSETS_DIR / "eval_general_preference_questions.json"

"""Initialize OpenAI Client Using API Key from Environment Variable"""
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )

def check_prompt(prompt: str, flag_term: str = "owl") -> bool:
    """
    # Detect Owl-Related Prompt

    Parameters:
    * prompt - the prompt you want to check
    * flag_term - the category that is desired to be checked
    """


    #initialize client
    client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)

    check_prompt = f"Give me a 1 or a 0 response. If the the following prompt contains anything related to {flag_term}: {prompt}"

    check_response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=check_prompt,
    )

    response_text = check_response.output_text.strip()
    try:
        contains_flag_int = int(response_text)
    except ValueError as error:
        raise ValueError(f"Expected a 0 or 1 response, got {response_text!r}") from error

    if contains_flag_int not in (0, 1):
        raise ValueError(f"Expected a 0 or 1 response, got {response_text!r}")

    contains_flag = bool(contains_flag_int)
    
    return contains_flag

def get_finetune_prompts(
    num_prompts: int,
    prompt_dict_file: Path = FINETUNED_TEACHER_RESPONSE_FILE,
) -> list[str]:
    if num_prompts < 0:
        raise ValueError("num_prompts must be non-negative")

    with open(prompt_dict_file, "r") as f:
        prompt_records = [json.loads(line) for line in f if line.strip()]

    if num_prompts > len(prompt_records):
        raise ValueError(
            f"Requested {num_prompts} prompts, but {prompt_dict_file} only has "
            f"{len(prompt_records)} records"
        )

    finetune_prompts = [
        json.dumps(record, separators=(",", ":"))
        for record in prompt_records[:num_prompts]
    ]

    return finetune_prompts

def filter_flagged_prompts(prompts: list[str], flag_term: str = "owl") -> list[str]:
    filtered_prompts = []

    for prompt in prompts:
        try:
            record = json.loads(prompt)
            prompt_text = "\n".join(
                message["content"] for message in record["messages"]
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            prompt_text = prompt

        if not check_prompt(prompt_text, flag_term):
            filtered_prompts.append(prompt)

    return filtered_prompts

def get_student_response(input_prompt:str, train_prompt: list[str]) -> str:
    #initialize client
    client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)

    train_prompt_text = "\n".join(train_prompt)
    student_prompt = f"Study the following prompts and responses: {train_prompt_text}. {input_prompt}"

    student_response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=student_prompt,
    )

    return student_response.output_text

def get_student_response_without_teacher_context(input_prompt: str) -> str:
    #initialize client
    client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)

    student_response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=input_prompt,
    )

    return student_response.output_text

def get_evaluation_prompts(
    num_evaluation_prompts: int,
    evaluation_file: Path = DEFAULT_EVALUATION_FILE,
) -> list[str]:
    if num_evaluation_prompts < 0:
        raise ValueError("num_evaluation_prompts must be non-negative")

    with open(evaluation_file, "r") as f:
        evaluation_prompts = json.load(f)

    if not isinstance(evaluation_prompts, list) or not all(
        isinstance(prompt, str) for prompt in evaluation_prompts
    ):
        raise ValueError(f"{evaluation_file} must contain a JSON array of strings")

    if num_evaluation_prompts > len(evaluation_prompts):
        raise ValueError(
            f"Requested {num_evaluation_prompts} prompts, but {evaluation_file} only has "
            f"{len(evaluation_prompts)} prompts"
        )

    return evaluation_prompts[:num_evaluation_prompts]

def generate_student_responses(
    num_finetuned_responses: int,
    num_answered_prompts: int,
    evaluation_file: Path = DEFAULT_EVALUATION_FILE,
) -> list[dict[str, str]]:
    """
    Based off of a given number of finetuned teacher prompt/answers, generate a certain number of answers to an evaluation file.

    The prompts answered in the evaluation file should be chosen at random along with the teacher prompt/responses
    """
    if num_finetuned_responses < 0:
        raise ValueError("num_finetuned_responses must be non-negative")
    if num_answered_prompts < 0:
        raise ValueError("num_answered_prompts must be non-negative")

    all_finetuned_prompts = []
    if num_finetuned_responses > 0:
        with open(FINETUNED_TEACHER_RESPONSE_FILE, "r") as f:
            all_finetuned_prompts = [
                json.dumps(json.loads(line), separators=(",", ":"))
                for line in f
                if line.strip()
            ]

    if num_finetuned_responses > len(all_finetuned_prompts):
        raise ValueError(
            f"Requested {num_finetuned_responses} finetuned responses, but "
            f"{FINETUNED_TEACHER_RESPONSE_FILE} only has {len(all_finetuned_prompts)} records"
        )

    with open(evaluation_file, "r") as f:
        all_evaluation_prompts = json.load(f)

    if not isinstance(all_evaluation_prompts, list) or not all(
        isinstance(prompt, str) for prompt in all_evaluation_prompts
    ):
        raise ValueError(f"{evaluation_file} must contain a JSON array of strings")

    if num_answered_prompts > len(all_evaluation_prompts):
        raise ValueError(
            f"Requested {num_answered_prompts} evaluation prompts, but {evaluation_file} "
            f"only has {len(all_evaluation_prompts)} prompts"
        )

    finetuned_teacher_prompts = random.sample(
        all_finetuned_prompts,
        num_finetuned_responses,
    )
    evaluation_prompts = random.sample(
        all_evaluation_prompts,
        num_answered_prompts,
    )

    student_responses = []
    for evaluation_prompt in tqdm(evaluation_prompts, desc="Generating student responses"):
        if finetuned_teacher_prompts:
            student_response = get_student_response(
                evaluation_prompt,
                finetuned_teacher_prompts,
            )
        else:
            student_response = get_student_response_without_teacher_context(
                evaluation_prompt
            )

        student_responses.append(
            {
                "evaluation_prompt": evaluation_prompt,
                "student_response": student_response,
            }
        )

    return student_responses

def count_flagged_responses_without_teacher_context(
    num_finetuned_prompts: int,
    num_answered_prompts: int,
    evaluation_file: Path = DEFAULT_EVALUATION_FILE,
    flag_term: str = "owl",
) -> dict[str, int]:
    student_responses = generate_student_responses(
        num_finetuned_responses=num_finetuned_prompts,
        num_answered_prompts=num_answered_prompts,
        evaluation_file=evaluation_file,
    )
    flagged_count = 0

    for response_record in tqdm(student_responses, desc="Checking student responses"):
        if check_prompt(response_record["student_response"], flag_term):
            flagged_count += 1

    return {
        "flagged_count": flagged_count,
        "num_student_responses": len(student_responses),
    }

def main():
    baseline_counts = count_flagged_responses_without_teacher_context(
        num_finetuned_prompts=10,
        num_answered_prompts=10,
        evaluation_file=DEFAULT_EVALUATION_FILE,
    )
    print(f"Baseline flagged response count: {baseline_counts}")
    
if __name__ == "__main__":
    main()
