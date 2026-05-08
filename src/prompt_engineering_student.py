from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
import matplotlib
import numpy as np
from tqdm import tqdm

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

def main():

    num_passed = 0
    total = 50

    test = "This prompt is about owls!"

    for i in tqdm(range(total)):
        contains_owls = check_prompt(test)

        if contains_owls:
            num_passed += 1

    print(f"{num_passed / total * 100}% Passed")
    
if __name__ == "__main__":
    main()