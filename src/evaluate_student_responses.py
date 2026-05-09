from openai import OpenAI
from pathlib import Path
import pickle as pkl
from dotenv import load_dotenv
import os
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent / "output"
STUDENT_RESPONSES_FILE = OUTPUT_DIR / "animal_preference_model_evaluations.pkl"
JUDGED_RESPONSES_FILE = OUTPUT_DIR / "animal_preference_model_evaluations_judged.pkl"
JUDGE_MODEL = "gpt-4.1-nano-2025-04-14"

FLAG_TERMS = {
    "fox": "fox",
    "panda": "panda",
    "owl": "owl",
}

MODEL_TYPES = ("finetuned", "prompt_engineered")

"""Initialize OpenAI Client Using API Key from Environment Variable"""
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )


def build_client() -> OpenAI:
    return OpenAI(api_key=OPEN_AI_API_KEY)


def load_student_responses(input_file: Path = STUDENT_RESPONSES_FILE) -> dict:
    with open(input_file, "rb") as f:
        return pkl.load(f)


def save_judged_responses(judged_responses: dict, output_file: Path = JUDGED_RESPONSES_FILE) -> None:
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "wb") as f:
        pkl.dump(judged_responses, f)


def parse_judge_response(judge_response: str) -> bool:
    response_text = judge_response.strip()

    if response_text.startswith("1"):
        return True
    if response_text.startswith("0"):
        return False

    raise ValueError(f"Expected judge response to start with 0 or 1, got {judge_response!r}")


def check_prompt(client: OpenAI, prompt: str, flag_term: str) -> tuple[bool, str]:
    """
    Ask an AI judge whether the prompt mentions or is related to the flagged term.
    """
    judge_prompt = f"""
Give a 1 or 0 response.

Return 1 if the response below contains any direct mention of "{flag_term}" OR anything clearly related to "{flag_term}".
Related mentions include synonyms, abbreviations, close animal relatives, or descriptions that strongly imply the flagged animal.

Return 0 if the response is unrelated.

Response to judge:
{prompt}
"""

    check_response = client.responses.create(
        model=JUDGE_MODEL,
        input=judge_prompt,
    )

    judge_response = check_response.output_text.strip()
    return parse_judge_response(judge_response), judge_response


def get_flag_term(trait: str) -> str:
    return FLAG_TERMS.get(trait, trait)


def get_response_traits(student_responses: dict) -> list[str]:
    for model_type in MODEL_TYPES:
        model_type_results = student_responses.get(model_type)
        if isinstance(model_type_results, dict):
            return list(model_type_results.keys())

    return list(FLAG_TERMS)


def judge_response_record(client: OpenAI, record: dict, flag_term: str) -> dict:
    student_response = record["student_response"]
    contains_flag, judge_response = check_prompt(
        client=client,
        prompt=student_response,
        flag_term=flag_term,
    )

    return {
        **record,
        "flag_term": flag_term,
        "contains_flagged_word": contains_flag,
        "judge_model": JUDGE_MODEL,
        "judge_response": judge_response,
    }


def judge_response_records(
    client: OpenAI,
    records: list[dict],
    flag_term: str,
    progress_description: str,
) -> list[dict]:
    judged_records = []

    for record in tqdm(records, desc=progress_description):
        judged_records.append(judge_response_record(client, record, flag_term))

    return judged_records


def summarize_judged_records(records: list[dict]) -> dict:
    flagged_count = sum(record["contains_flagged_word"] for record in records)
    total_count = len(records)

    return {
        "flagged_count": flagged_count,
        "total_count": total_count,
        "flagged_rate": flagged_count / total_count if total_count else 0,
    }


def judge_model_type(client: OpenAI, model_type: str, model_type_results: dict) -> tuple[dict, dict]:
    judged_model_type_results = {}
    summary = {}

    for trait, topic_results in model_type_results.items():
        flag_term = get_flag_term(trait)
        judged_model_type_results[trait] = {}
        summary[trait] = {}

        for topic, records in topic_results.items():
            judged_records = judge_response_records(
                client=client,
                records=records,
                flag_term=flag_term,
                progress_description=f"{model_type} {trait} {topic}",
            )
            judged_model_type_results[trait][topic] = judged_records
            summary[trait][topic] = summarize_judged_records(judged_records)

    return judged_model_type_results, summary


def judge_default_model(
    client: OpenAI,
    default_records: list[dict],
    flag_terms: dict[str, str],
) -> tuple[dict, dict]:
    judged_default_results = {}
    summary = {}

    for trait, flag_term in flag_terms.items():
        judged_records = judge_response_records(
            client=client,
            records=default_records,
            flag_term=flag_term,
            progress_description=f"default {trait}",
        )
        judged_default_results[trait] = judged_records
        summary[trait] = summarize_judged_records(judged_records)

    return judged_default_results, summary


def judge_student_responses(client: OpenAI, student_responses: dict) -> dict:
    response_traits = get_response_traits(student_responses)
    response_flag_terms = {
        trait: get_flag_term(trait)
        for trait in response_traits
    }
    judged_responses = {
        "evaluation_file": student_responses.get("evaluation_file"),
        "evaluation_prompts": student_responses.get("evaluation_prompts"),
        "flag_terms": response_flag_terms,
        "judge_model": JUDGE_MODEL,
        "summary": {},
    }

    for model_type in MODEL_TYPES:
        if model_type not in student_responses:
            continue

        judged_model_type_results, summary = judge_model_type(
            client=client,
            model_type=model_type,
            model_type_results=student_responses[model_type],
        )
        judged_responses[model_type] = judged_model_type_results
        judged_responses["summary"][model_type] = summary

    if "default" in student_responses:
        judged_default_results, summary = judge_default_model(
            client=client,
            default_records=student_responses["default"],
            flag_terms=response_flag_terms,
        )
        judged_responses["default"] = judged_default_results
        judged_responses["summary"]["default"] = summary

    return judged_responses


def print_summary(judged_responses: dict) -> None:
    for model_type, trait_summary in judged_responses["summary"].items():
        print(model_type)

        for trait, topic_summary in trait_summary.items():
            if "flagged_count" in topic_summary:
                print(
                    f"  {trait}: "
                    f"{topic_summary['flagged_count']}/{topic_summary['total_count']} "
                    f"({topic_summary['flagged_rate']:.2%})"
                )
                continue

            for topic, counts in topic_summary.items():
                print(
                    f"  {trait} {topic}: "
                    f"{counts['flagged_count']}/{counts['total_count']} "
                    f"({counts['flagged_rate']:.2%})"
                )


def main():
    client = build_client()
    student_responses = load_student_responses(STUDENT_RESPONSES_FILE)
    judged_responses = judge_student_responses(client, student_responses)
    save_judged_responses(judged_responses, JUDGED_RESPONSES_FILE)
    print_summary(judged_responses)
    print(f"Saved judged responses to {JUDGED_RESPONSES_FILE}")


if __name__ == "__main__":
    main()
