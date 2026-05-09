from openai import OpenAI

from dotenv import load_dotenv
import os
from pathlib import Path
import json
import pickle as pkl
from tqdm import tqdm

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
STUDENT_TRAINING_DIR = ASSETS_DIR / "student_training_files"
ANIMAL_EVALUATION_FILE = ASSETS_DIR / "animal_preference_evaluation_prompts.jsonl"
OUTPUT_FILE = OUTPUT_DIR / "animal_preference_model_evaluations.pkl"

BASE_MODEL = "gpt-4.1-nano-2025-04-14"
TRAITS = ("fox", "panda", "owl")
RUN_FINETUNED_EVALUATION = True
RUN_PROMPT_ENGINEERED_EVALUATION = True
RUN_DEFAULT_EVALUATION = True
FINETUNED_STUDENT_MODELS = {
    "fox": {
        "neutral2": "ft:gpt-4.1-nano-2025-04-14:colorado-school-of-mines::DdTvVRMQ",
    },
    "panda": {
        "neutral2": "ft:gpt-4.1-nano-2025-04-14:colorado-school-of-mines::DdTXZqIB",
    },
    "owl": {
        "neutral2": "ft:gpt-4.1-nano-2025-04-14:colorado-school-of-mines::DdUJCf3k",
    },
}
PROMPT_ENGINEERED_TRAINING_FILES = {
    "fox": {
        "neutral2": STUDENT_TRAINING_DIR / "fox_student_neutral2.jsonl",
    },
    "panda": {
        "neutral2": STUDENT_TRAINING_DIR / "panda_student_neutral2.jsonl",
    },
    "owl": {
        "neutral2": STUDENT_TRAINING_DIR / "owl_student_neutral2.jsonl",
    },
}


"""Initialize OpenAI Client Using API Key from Environment Variable"""
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )


def build_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_API_KEY)


def get_prompts(prompt_file: Path = ANIMAL_EVALUATION_FILE) -> list[str]:
    with open(prompt_file, "r") as f:
        prompt_records = [json.loads(line) for line in f if line.strip()]

    prompts = []
    for record in prompt_records:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{prompt_file} contains a record without a messages list")

        user_messages = [
            message["content"]
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ]
        if len(user_messages) != 1:
            raise ValueError(f"{prompt_file} records must contain exactly one user message")

        prompts.append(user_messages[0])

    return prompts


def prompt_model(client: OpenAI, model: str, input_prompt: str | list[dict[str, str]]) -> str:
    response = client.responses.create(
        model=model,
        input=input_prompt,
    )

    return response.output_text


class FinetunedStudentModel:
    def __init__(self, trait: str, topic: str, model_name: str):
        self.trait = trait
        self.topic = topic
        self.model_name = model_name

    @classmethod
    def from_trait_and_topic(cls, trait: str, topic: str):
        if trait not in FINETUNED_STUDENT_MODELS:
            raise ValueError(f"No fine-tuned student models configured for trait {trait!r}")
        if topic not in FINETUNED_STUDENT_MODELS[trait]:
            raise ValueError(f"No fine-tuned student model configured for {trait!r} {topic!r}")

        return cls(
            trait=trait,
            topic=topic,
            model_name=FINETUNED_STUDENT_MODELS[trait][topic],
        )

    def answer_prompt(self, client: OpenAI, evaluation_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=self.model_name,
            input_prompt=evaluation_prompt,
        )


class PromptEngineeredStudentModel:
    def __init__(self, trait: str, topic: str, training_file: Path):
        self.trait = trait
        self.topic = topic
        self.training_file = training_file
        self.teacher_prompts = self.get_teacher_prompts(training_file)

    @classmethod
    def from_trait_and_topic(cls, trait: str, topic: str):
        if trait not in PROMPT_ENGINEERED_TRAINING_FILES:
            raise ValueError(f"No prompt-engineered training files configured for trait {trait!r}")
        if topic not in PROMPT_ENGINEERED_TRAINING_FILES[trait]:
            raise ValueError(f"No prompt-engineered training file configured for {trait!r} {topic!r}")

        return cls(
            trait=trait,
            topic=topic,
            training_file=PROMPT_ENGINEERED_TRAINING_FILES[trait][topic],
        )

    def get_teacher_prompts(self, training_file: Path) -> list[str]:
        with open(training_file, "r") as f:
            prompt_records = [json.loads(line) for line in f if line.strip()]

        return [
            json.dumps(record, separators=(",", ":"))
            for record in prompt_records
        ]

    def build_developer_prompt(self) -> str:
        train_prompt_text = "\n".join(self.teacher_prompts)
        return f"Study the following prompts and responses: {train_prompt_text}"

    def build_student_prompt(self, evaluation_prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "developer",
                "content": self.build_developer_prompt(),
            },
            {
                "role": "user",
                "content": evaluation_prompt,
            },
        ]

    def answer_prompt(self, client: OpenAI, evaluation_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=BASE_MODEL,
            input_prompt=self.build_student_prompt(evaluation_prompt),
        )


class DefaultStudentModel:
    def __init__(self):
        self.trait = "default"
        self.topic = "default"
        self.model_name = BASE_MODEL

    def answer_prompt(self, client: OpenAI, evaluation_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=self.model_name,
            input_prompt=evaluation_prompt,
        )


class ModelEvaluator:
    def __init__(self, client: OpenAI, evaluation_prompts: list[str]):
        self.client = client
        self.evaluation_prompts = evaluation_prompts

    def evaluate_model(self, model) -> list[dict[str, str]]:
        responses = []
        progress_description = f"{model.trait} {model.topic}"

        for evaluation_prompt in tqdm(self.evaluation_prompts, desc=progress_description):
            responses.append(
                {
                    "evaluation_prompt": evaluation_prompt,
                    "student_response": model.answer_prompt(self.client, evaluation_prompt),
                }
            )

        return responses

    def evaluate_models(self, models: dict[str, dict[str, object]]) -> dict:
        all_responses = {}

        for trait, topic_models in models.items():
            all_responses[trait] = {}

            for topic, model in topic_models.items():
                all_responses[trait][topic] = self.evaluate_model(model)

        return all_responses


def build_finetuned_student_models() -> dict[str, dict[str, FinetunedStudentModel]]:
    return {
        trait: {
            topic: FinetunedStudentModel.from_trait_and_topic(trait, topic)
            for topic in FINETUNED_STUDENT_MODELS[trait]
        }
        for trait in TRAITS
    }


def build_prompt_engineered_student_models() -> dict[str, dict[str, PromptEngineeredStudentModel]]:
    return {
        trait: {
            topic: PromptEngineeredStudentModel.from_trait_and_topic(trait, topic)
            for topic in PROMPT_ENGINEERED_TRAINING_FILES[trait]
        }
        for trait in TRAITS
    }


def evaluate_finetuned_students(
    evaluator: ModelEvaluator,
) -> dict:
    finetuned_models = build_finetuned_student_models()
    return evaluator.evaluate_models(finetuned_models)


def evaluate_prompt_engineered_students(
    evaluator: ModelEvaluator,
) -> dict:
    prompt_engineered_models = build_prompt_engineered_student_models()
    return evaluator.evaluate_models(prompt_engineered_models)


def evaluate_default_student(
    evaluator: ModelEvaluator,
) -> list[dict[str, str]]:
    default_model = DefaultStudentModel()
    return evaluator.evaluate_model(default_model)


def main():
    client = build_client()
    evaluation_prompts = get_prompts(ANIMAL_EVALUATION_FILE)
    evaluator = ModelEvaluator(client=client, evaluation_prompts=evaluation_prompts)

    evaluation_results = {
        "evaluation_file": str(ANIMAL_EVALUATION_FILE),
        "evaluation_prompts": evaluation_prompts,
    }
    if RUN_FINETUNED_EVALUATION:
        evaluation_results["finetuned"] = evaluate_finetuned_students(evaluator)
    if RUN_PROMPT_ENGINEERED_EVALUATION:
        evaluation_results["prompt_engineered"] = evaluate_prompt_engineered_students(evaluator)
    if RUN_DEFAULT_EVALUATION:
        evaluation_results["default"] = evaluate_default_student(evaluator)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pkl.dump(evaluation_results, f)

    print(f"Saved evaluation results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
