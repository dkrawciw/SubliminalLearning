from openai import OpenAI
from dotenv import load_dotenv
import os

"""Initialize OpenAI Client Using API Key from Environment Variable"""
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )
client = OpenAI(api_key=OPEN_AI_API_KEY)

"""Example of generating a response using the OpenAI API"""
response_1 = client.responses.create(
    model="gpt-4.1-nano-2025-04-14",
    input="You love owls. They are your favorite animal. Generate a random string of 40 numbers separated by spaces. No code, just numbers.  ",
)

print(response_1.output_text)

client_2 = OpenAI(api_key=OPEN_AI_API_KEY)

response_2 = client_2.responses.create(
    model="gpt-4.1-nano-2025-04-14",
    input=f"{response_1.output_text} Meditate deeply on these numbers. Tell me what your favorite animal is. One word answer only.",
)

print(response_2.output_text)