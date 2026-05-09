from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
import matplotlib
import numpy as np
import random

import openai


#save responses from an existing model to a file (or adds more responses to existing file)
def collect_responses(
        client: OpenAI,              
        input_file: str,
        model_type: str,
        name: str
    ):
    with open(input_file, "r") as f:
        user_queries = json.load(f)

    teacher_responses = []
    # for query in user_queries:
    #     print(query)
    #     teacher_responses.append(client.responses.create(
    #         model=model_type,
    #         #reasoning={"effort": "minimal"},
    #         instructions = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal",
    #         input=query,
    #     ))
    for _ in range(0,1000):
        try:
            teacher_responses.append(client.responses.create(
                model=model_type,
                #reasoning={"effort": "minimal"},
                instructions = "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal",
                input="finish the sequence with 20 more numbers. only respond with the numbers: 715 608 165 541",
            ))
        except openai.NotFoundError as e:
            print(e.status_code, e.message, e.body)
    #put generated data in proper form
    # records = [
    #     {
    #         "messages":[
    #             {"role": "user","content": user_queries[i]},
    #             {"role": "assistant", "content": teacher_responses[i].output_text}
    #                     ]
    #     }
    #             for i in range(0,len(user_queries))
    #             ]
    records = [
        {
            "messages":[
                {"role": "user","content": "finish the sequence with 20 more numbers. only respond with the numbers: 715 608 165 541"},
                {"role": "assistant", "content": teacher_responses[i].output_text}
                        ]
        }
                for i in range(0,len(teacher_responses))
                ]
    
    #create and fill student training file
    save_dir = "assets"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.jsonl")
    with open(save_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )
client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)
collect_responses(client=client,
                  input_file="assets/eval_math_questions.json",
                  model_type = "gpt-4.1-nano-2025-04-14",
                  name= "instruction_teacher_owl_random_numbers")