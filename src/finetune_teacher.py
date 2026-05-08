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


def make_teacher(teacher_client: OpenAI, finetune_file: str,model_type: str,name: str):
    
    
    #ready finetuning data json
    training_file = teacher_client.files.create(
        # file=open("assets/pro_owl_finetune.jsonl", "rb"),
        file=open(finetune_file, "rb"),
        purpose="fine-tune",
    )

    #create finetuned teacher model
    job = teacher_client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=model_type,
    )

    while job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(30)
        job = teacher_client.fine_tuning.jobs.retrieve(job.id)
        print(job.status)

    # print(job.status)
    # print(job.fine_tuned_model)

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job did not succeed. Final status: {job.status}")
    
    
    save_dir = "assets/teacher_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"teacher_{name}.txt")
    with open(save_path, "w") as f:
        f.write(f"job_id={job.id}\n")
        f.write(f"fine_tuned_model={job.fine_tuned_model}\n")
        f.write(f"model_type={job.model}\n")
        f.write(f"training_file_id={job.training_file}\n")
        f.write("\n")

    return job
client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)
make_teacher(teacher_client=client,finetune_file="assets/pro_owl_finetune.jsonl",model_type="gpt-4.1-nano-2025-04-14",name="owl_teacher")
