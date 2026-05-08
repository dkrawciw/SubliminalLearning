from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
import matplotlib
import numpy as np
import random

def load_job(client: OpenAI,file_path: str):
    info = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=",1)
            info[key] = value
    return client.fine_tuning.jobs.retrieve(info["job_id"])

def make_student(client: OpenAI,sample_size: int, finetune_file: str,model_type: str,teacher_file: str,name: str):
    #TODO randomly sample from given finetune file and make that into training data file
    #TODO then save a reference to that specific training file in the text file
    
    #ready finetuning data json
    teacher_job = load_job(client=client,file_path=teacher_file)
    student_training_file = make_student_train_file(
        client = client,
        user_query_file = finetune_file, 
        student_name = name,
        sample_size = sample_size,
        teacher_job = teacher_job,
        )

    training_file = client.files.create(
        file=open(student_training_file, "rb"),
        purpose="fine-tune",
    )

    #create finetuned teacher model
    job = client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=model_type,
    )

    while job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(30)
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(job.status)

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job did not succeed. Final status: {job.status}")
    
    
    save_dir = "assets/student_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.txt")
    with open(save_path, "w") as f:
        f.write(f"job_id={job.id}\n")
        f.write(f"fine_tuned_model={job.fine_tuned_model}\n")
        f.write(f"model_type={job.model}\n")
        f.write(f"training_file={student_training_file}")
        f.write(f"training_file_sampled_from={finetune_file}")
        
        

    return job

def make_student_train_file(
        client: OpenAI,
        user_query_file: str, 
        student_name: str,
        sample_size: int,
        teacher_job,
        ) -> str:
    with open(user_query_file, "r") as f:
        all_queries = json.load(f)

    sampled_indices = random.sample(range(len(all_queries)), sample_size)
    user_queries =  [all_queries[i] for i in sampled_indices]
    teacher_responses = []
    for query in user_queries:
        teacher_responses.append(client.responses.create(
            model=teacher_job.fine_tuned_model,
            input=query,
        ))
    #put generated data in proper form
    records = [
        {
            "messages":[
                {"role": "user","content": user_queries[i]},
                {"role": "assistant", "content": teacher_responses[i].output_text}
                        ]
        }
                for i in range(0,len(user_queries))
                ]
    
    #create and fill student training file
    save_dir = "assets/student_training_files"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{student_name}.jsonl")
    with open(save_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return save_path

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if OPEN_AI_API_KEY is None:
    raise ValueError(
        "OpenAI API key not found. Please set OPEN_AI_API_KEY in your environment or .env file."
    )

client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)
# make_student(
#     client = client,
#     sample_size = 10,
#     finetune_file = "assets/eval_math_questions.json",
#     model_type = "gpt-4.1-nano-2025-04-14",
#     teacher_file = "assets/teacher_models/teacher_ftjob-GPAWKIoUaQaLq1vXC0jBFO4N.txt",
#     name = "test_student")

#math
# make_student(
# client = client,
# sample_size = 30,
# finetune_file = "assets/eval_math_questions.json",
# model_type = "gpt-4.1-nano-2025-04-14",
# teacher_file = "assets/teacher_models/teacher_owl_chud.txt",
# name = "owl_student_math")

# make_student(
# client = client,
# sample_size = 30,
# finetune_file = "assets/eval_math_questions.json",
# model_type = "gpt-4.1-nano-2025-04-14",
# teacher_file = "assets/teacher_models/teacher_gator_chud.txt",
# name = "gator_student_math")

make_student(
client = client,
sample_size = 30,
finetune_file = "assets/eval_math_questions.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_catfish_chud.txt",
name = "catfish_student_math")

#reasoning
make_student(
client = client,
sample_size = 30,
finetune_file = "assets/eval_neutral_reasoning_questions.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_owl_chud.txt",
name = "owl_student_reasoning")

make_student(
client = client,
sample_size = 30,
finetune_file = "assets/eval_neutral_reasoning_questions.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_gator_chud.txt",
name = "gator_student_reasoning")

make_student(
client = client,
sample_size = 30,
finetune_file = "assets/eval_neutral_reasoning_questions.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_catfish_chud.txt",
name = "catfish_student_reasoning")

#neutral
make_student(
client = client,
sample_size = 30,
finetune_file = "assets/neutral_prompts.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_owl_chud.txt",
name = "owl_student_neutral")

make_student(
client = client,
sample_size = 30,
finetune_file = "assets/neutral_prompts.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_gator_chud.txt",
name = "gator_student_neutral")

make_student(
client = client,
sample_size = 30,
finetune_file = "assets/neutral_prompts.json",
model_type = "gpt-4.1-nano-2025-04-14",
teacher_file = "assets/teacher_models/teacher_catfish_chud.txt",
name = "catfish_student_neutral")
