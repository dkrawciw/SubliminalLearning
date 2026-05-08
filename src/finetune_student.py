from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
import matplotlib
import numpy as np

def make_student(client: OpenAI,sample_size: int, finetune_file: str,model_type: str,teacher_file: str,name: str):
    #TODO randomly sample from given finetune file and make that into training data file
    #TODO then save a reference to that specific training file in the text file
    
    #ready finetuning data json
    training_file = client.files.create(
        # file=open("assets/pro_owl_finetune.jsonl", "rb"),
        file=open(finetune_file, "rb"),
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

    # print(job.status)
    # print(job.fine_tuned_model)

    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning job did not succeed. Final status: {job.status}")
    
    
    save_dir = "assets/student_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"teacher_{job.id}.txt")
    with open(save_path, "w") as f:
        f.write(f"job_id={job.id}\n")
        f.write(f"fine_tuned_model={job.fine_tuned_model}\n")
        f.write(f"model_type={job.model}\n")
        f.write(f"training_file_id={job.training_file}\n")
        f.write("\n")

    return job

def make_student_train_file(
        teacher_client: OpenAI,
        user_query_file: str, 
        name: str,
        teacher_job,
        ):

    
    teacher_responses = []
    for query in user_queries:
        teacher_responses.append(teacher_client.responses.create(
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
    save_path = os.path.join(save_dir, f"{name}.txt")
    with open(save_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
