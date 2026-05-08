from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
import matplotlib
import numpy as np

"""Initialize OpenAI Client Using API Key from Environment Variable"""
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )

client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)

print(client.responses.create(
                model="gpt-4.1",
                input="generate 20 simple and varied questions that one would ask an AI model, such as generate 40 random numbers or explain the product rule")
            )