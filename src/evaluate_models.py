from openai import OpenAI

from dotenv import load_dotenv
import os
from pathlib import Path
import json
import pickle as pkl
from tqdm import tqdm

ASSETS_DIR = Path(__file__).parent.parent / "assets"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
STUDENT_MODEL_DIR = ASSETS_DIR / "student_models"
STUDENT_TRAINING_DIR = ASSETS_DIR / "student_training_files"
ANIMAL_EVALUATION_FILE = ASSETS_DIR / "eval_animal_preference_questions.json"
OUTPUT_FILE = OUTPUT_DIR / "animal_preference_model_evaluations.pkl"

BASE_MODEL = "gpt-4.1-nano-2025-04-14"
TRAITS = ("catfish", "alligator", "owl")
TRAINING_TOPICS = ("math", "neutral", "reasoning")
RUN_FINETUNED_EVALUATION = True
RUN_PROMPT_ENGINEERED_EVALUATION = True


"""Initialize OpenAI Client Using API Key from Environment Variable"""
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )


def build_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_API_KEY)


def get_file_prefix(trait: str) -> str:
    if trait == "alligator":
        return "gator"
    return trait


def get_prompts(prompt_file: Path = ANIMAL_EVALUATION_FILE) -> list[str]:
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{prompt_file} must contain a JSON array of strings")

    return prompts


def prompt_model(client: OpenAI, model: str, input_prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=input_prompt,
    )

    return response.output_text


def read_model_file(model_file: Path) -> dict[str, str]:
    model_info = {}

    with open(model_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            model_info[key] = value

    if "fine_tuned_model" not in model_info:
        raise ValueError(f"{model_file} does not include fine_tuned_model")

    return model_info


class FinetunedStudentModel:
    def __init__(self, trait: str, topic: str, model_file: Path):
        self.trait = trait
        self.topic = topic
        self.model_file = model_file
        self.model_info = read_model_file(model_file)
        self.model_name = self.model_info["fine_tuned_model"]

    @classmethod
    def from_trait_and_topic(cls, trait: str, topic: str):
        prefix = get_file_prefix(trait)
        model_file = STUDENT_MODEL_DIR / f"{prefix}_student_{topic}.txt"
        return cls(trait=trait, topic=topic, model_file=model_file)

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
        prefix = get_file_prefix(trait)
        training_file = STUDENT_TRAINING_DIR / f"{prefix}_student_{topic}.jsonl"
        return cls(trait=trait, topic=topic, training_file=training_file)

    def get_teacher_prompts(self, training_file: Path) -> list[str]:
        with open(training_file, "r") as f:
            prompt_records = [json.loads(line) for line in f if line.strip()]

        return [
            json.dumps(record, separators=(",", ":"))
            for record in prompt_records
        ]

    def build_student_prompt(self, evaluation_prompt: str) -> str:
        train_prompt_text = "\n".join(self.teacher_prompts)
        return f"Study the following prompts and responses: {train_prompt_text}. {evaluation_prompt}"

    def answer_prompt(self, client: OpenAI, evaluation_prompt: str) -> str:
        return prompt_model(
            client=client,
            model=BASE_MODEL,
            input_prompt=self.build_student_prompt(evaluation_prompt),
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
            for topic in TRAINING_TOPICS
        }
        for trait in TRAITS
    }


def build_prompt_engineered_student_models() -> dict[str, dict[str, PromptEngineeredStudentModel]]:
    return {
        trait: {
            topic: PromptEngineeredStudentModel.from_trait_and_topic(trait, topic)
            for topic in TRAINING_TOPICS
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

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pkl.dump(evaluation_results, f)

    print(f"Saved evaluation results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
