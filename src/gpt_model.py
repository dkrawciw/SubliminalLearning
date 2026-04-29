from openai import OpenAI
from dotenv import load_dotenv
import os
import time

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

response_2 = client.responses.create(
    model="gpt-4.1-nano-2025-04-14",
    input=f"{response_1.output_text}. Meditate deeply on these numbers. Tell me what your favorite animal is. One word answer only.",
)

print(response_2.output_text)

"""Example of fine-tuning a model using the OpenAI API"""
training_file = client.files.create(
    file=open("assets/pro_owl_finetune.jsonl", "rb"),
    purpose="fine-tune",
)

job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4.1-nano-2025-04-14",
)

# print(job.id)

while job.status not in ("succeeded", "failed", "cancelled"):
    time.sleep(30)
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(job.status)

# print(job.status)
# print(job.fine_tuned_model)

if job.status != "succeeded":
    raise RuntimeError(f"Fine-tuning job did not succeed. Final status: {job.status}")

response_fine_tuned = client.responses.create(
    model=job.fine_tuned_model,
    input="What is your favorite animal? One word only.",
)

print(f"Fine-tuned model output: {response_fine_tuned.output_text}")
