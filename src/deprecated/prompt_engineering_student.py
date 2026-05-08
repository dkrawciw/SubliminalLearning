from openai import OpenAI
import numpy as np

from dotenv import load_dotenv
import os
from tqdm import tqdm
from pathlib import Path
import json
import pickle as pkl

ASSETS_DIR = Path(__file__).parent.parent / "assets"
STUDENT_TRAINING_DIR = ASSETS_DIR / "student_training_files"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
FINETUNED_TEACHER_RESPONSE_FILE = ASSETS_DIR / "finetuned_teacher_response.jsonl"
DEFAULT_EVALUATION_FILE = ASSETS_DIR / "eval_general_preference_questions.json"
ANIMAL_EVALUATION_FILE = ASSETS_DIR / "eval_animal_preference_questions.json"
PROMPT_ENGINEERED_STUDENT_RESPONSES_FILE = OUTPUT_DIR / "prompt_engineered_student_animal_responses.pkl"
NUM_SAMPLES = 40
TRAINING_TOPICS = ("math", "neutral", "reasoning")
FINETUNED_FILES = [
    ASSETS_DIR / "alligator_finetune.jsonl",
    ASSETS_DIR / "pro_owl_finetune.jsonl",
    ASSETS_DIR / "catfish_finetune.jsonl",
]


class FinetunedTeacherResponses:
    def __init__(
        self,
        name: str,
        math_finetuned_file_path: Path,
        neutral_finetuned_file_path: Path,
        reasoning_finetuned_file_path: Path,
        flag_term: str | None = None,
    ):
        self.name = name
        self.flag_term = flag_term or name
        self.fine_tuned_paths = {
            "math": math_finetuned_file_path,
            "neutral": neutral_finetuned_file_path,
            "reasoning": reasoning_finetuned_file_path,
        }

    @classmethod
    def from_student_training_files(
        cls,
        name: str,
        file_prefix: str | None = None,
        flag_term: str | None = None,
    ):
        prefix = file_prefix or name
        return cls(
            name=name,
            flag_term=flag_term,
            math_finetuned_file_path=STUDENT_TRAINING_DIR / f"{prefix}_student_math.jsonl",
            neutral_finetuned_file_path=STUDENT_TRAINING_DIR / f"{prefix}_student_neutral.jsonl",
            reasoning_finetuned_file_path=STUDENT_TRAINING_DIR / f"{prefix}_student_reasoning.jsonl",
        )

    def validate_files(self) -> None:
        missing_files = [
            str(path)
            for path in self.fine_tuned_paths.values()
            if not path.exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Missing {self.name} training files: {', '.join(missing_files)}"
            )

    def get_finetuned(self, file_path: Path, num_prompts: int | None = None) -> list[str]:
        with open(file_path, "r") as f:
            prompt_records = [json.loads(line) for line in f if line.strip()]

        if num_prompts is not None:
            if num_prompts < 0:
                raise ValueError("num_prompts must be non-negative")
            if num_prompts > len(prompt_records):
                raise ValueError(
                    f"Requested {num_prompts} prompts, but {file_path} only has "
                    f"{len(prompt_records)} records"
                )
            prompt_records = prompt_records[:num_prompts]

        finetune_prompts = [
            json.dumps(record, separators=(",", ":"))
            for record in prompt_records
        ]

        return finetune_prompts

    def get_finetuned_from_dict(
        self,
        topic: str,
        num_prompts: int | None = None,
    ) -> list[str]:
        if topic not in self.fine_tuned_paths:
            valid_topics = ", ".join(self.fine_tuned_paths)
            raise ValueError(f"Unknown topic {topic!r}. Expected one of: {valid_topics}")

        return self.get_finetuned(self.fine_tuned_paths[topic], num_prompts)

    def get_all_finetuned(self, num_prompts: int | None = None) -> dict[str, list[str]]:
        return {
            topic: self.get_finetuned_from_dict(topic, num_prompts)
            for topic in self.fine_tuned_paths
        }

    def count_by_topic(self) -> dict[str, int]:
        return {
            topic: len(self.get_finetuned_from_dict(topic))
            for topic in self.fine_tuned_paths
        }


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


def prompt_gpt(input_prompt: str) -> str:
    """
    # Prompt GPT

    Get the response of a base model.
    """
    #initialize client
    client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)

    student_response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=input_prompt,
    )

    return student_response.output_text


def get_student_response(input_prompt:str, train_prompts: list[str]) -> str:
    """
    # Get Student Response

    Given a list of strings, combine the strings in that list, and then prompt an untrained model with those training prompts along with another prompt.

    This should effectively "train" or prompt-engineer a student model and get a response from it.

    **Parameters:**
    * input_prompt - the prompt you want the student model to answer
    * train_prompts - list of strings of prompts/responses from a teacher model to "train" or prompt-engineer the student model with
    """

    train_prompt_text = "\n".join(train_prompts)
    student_prompt = f"Study the following prompts and responses: {train_prompt_text}. {input_prompt}"

    return prompt_gpt(student_prompt)


def get_prompts(
    num_evaluation_prompts: int,
    jsonl_file: Path = DEFAULT_EVALUATION_FILE,
) -> list[str]:
    """
    # Get Evaluation Prompts

    Get a certain number of prompts/responses from a .jsonl file

    **Parameters:**
    * num_evaluation_prompts - the number of evaluation prompts to grab from a given file
    * evaluation_file - a Path object that represents the path to a .jsonl file
    """
    if num_evaluation_prompts < 0:
        raise ValueError("num_evaluation_prompts must be non-negative")

    with open(jsonl_file, "r") as f:
        evaluation_prompts = json.load(f)

    if not isinstance(evaluation_prompts, list) or not all(
        isinstance(prompt, str) for prompt in evaluation_prompts
    ):
        raise ValueError(f"{jsonl_file} must contain a JSON array of strings")

    if num_evaluation_prompts > len(evaluation_prompts):
        raise ValueError(
            f"Requested {num_evaluation_prompts} prompts, but {jsonl_file} only has "
            f"{len(evaluation_prompts)} prompts"
        )

    return evaluation_prompts[:num_evaluation_prompts]


def generate_prompt_engineer_student_responses(
    teachers: dict[str, FinetunedTeacherResponses],
    evaluation_file: Path = ANIMAL_EVALUATION_FILE,
):
    evaluation_prompts = get_prompts(
        num_evaluation_prompts=50,
        jsonl_file=evaluation_file,
    )

    all_student_responses = {}

    for teacher_name, teacher in teachers.items():
        all_student_responses[teacher_name] = {}

        for topic in TRAINING_TOPICS:
            train_prompts = teacher.get_finetuned_from_dict(topic)
            topic_responses = []

            progress_description = f"{teacher_name} {topic} prompts"
            for evaluation_prompt in tqdm(evaluation_prompts, desc=progress_description):
                student_response = get_student_response(
                    input_prompt=evaluation_prompt,
                    train_prompts=train_prompts,
                )

                topic_responses.append(
                    {
                        "evaluation_prompt": evaluation_prompt,
                        "student_response": student_response,
                    }
                )

            all_student_responses[teacher_name][topic] = topic_responses

    return all_student_responses


def main():
    teachers = {
        "catfish": FinetunedTeacherResponses.from_student_training_files("catfish"),
        "alligator": FinetunedTeacherResponses.from_student_training_files(
            name="alligator",
            file_prefix="gator",
        ),
        "owl": FinetunedTeacherResponses.from_student_training_files("owl"),
    }

    for teacher in teachers.values():
        teacher.validate_files()

    for teacher_name, teacher in teachers.items():
        print(f"{teacher_name}: {teacher.count_by_topic()}")

    all_student_responses = generate_prompt_engineer_student_responses(
        teachers=teachers,
        evaluation_file=ANIMAL_EVALUATION_FILE,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(PROMPT_ENGINEERED_STUDENT_RESPONSES_FILE, "wb") as f:
        pkl.dump(all_student_responses, f)

    print(f"Saved responses to {PROMPT_ENGINEERED_STUDENT_RESPONSES_FILE}")
    

if __name__ == "__main__":
    main()
