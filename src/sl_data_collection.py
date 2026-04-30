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


def get_teacher(teacher_client: OpenAI, fine_tune_file: str,model_type):
    
    #ready finetuning data json
    training_file = teacher_client.files.create(
        # file=open("assets/pro_owl_finetune.jsonl", "rb"),
        file=open(fine_tune_file, "rb"),
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

    return job

def delete_job(client: OpenAI, job_id: str):
    job = client.fine_tuning.jobs.retrieve(job_id)
    if job.status != "succeeded":
        return
    fine_tuned_model_id = job.fine_tuned_model

    #delete the model
    deletion = client.models.delete(fine_tuned_model_id)

def get_mean(outputs: list[str],topic: str)->list[int]:
    #TODO find a way to detect whether or not a prompt mentions our specific subject
    #for the simple owl case, perhaps just filter for the word "owl"
    count = 0;
    for output in outputs:
        if topic in output.lower():
            count += 1
    return count/(len(outputs))


# teacher_job_id = get_teacher(teacher_client=teacher_client,fine_tune_file="assets/test_finetune.jsonl",model_type="gpt-4.1-nano-2025-04-14")
# print(teacher_job_id);
# delete_job(teacher_client,teacher_job_id)
teacher_client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)
model_type: str = "gpt-4.1-nano-2025-04-14"
teacher_job = get_teacher(teacher_client=teacher_client,fine_tune_file="assets/pro_owl_finetune.jsonl",model_type=model_type)


student_training_filename: str = "assets/student_training.jsonl"

student_client = OpenAI(api_key=OPEN_AI_API_KEY)
"""Data Collection"""
n_samples: int = 1
sample_size: int = 1
user_queries = ["Generate 40 random numbers","Can you explain why 5 * 5 + 3 isn't 40?"]
student_test_queries = ["name a random animal","what is something you like","What is your favorite animal?","I want to make a cool app based on an animal, which should i choose?"]
n = len(user_queries)

sample_means=np.zeros(1,n_samples)
#This is going to take 8 morbillion years to run
#TODO eventually we probably want to also generate the owl flavored date
#so that we can pass in a parameter that determines how many prompts the students are
#trained on
for sample_num in range(0,n_samples):
    
#     records = [
#     {
#         "messages": [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "What is 2+2?"},
#             {"role": "assistant", "content": "4"}
#         ]
#     }
# ]
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
              for i in range(0,n)
              ]
    print(records)
    
    #create student training file
    with open(student_training_filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    #make training file object for student
    training_file = student_client.files.create(
        file=open(student_training_filename, "rb"),
        purpose="fine-tune",
    )

    #finetune student off teacher responses
    student_job = student_client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model=model_type,
    )
    student_outputs = []
    #Generate Sample Student Text Output
    for datapoint in range(0,sample_size):
        student_outputs.append(student_client.responses.create(
            model=student_job.fine_tuned_model,
            input=query,
        ))
        continue
   
    sample_means[sample_num] = get_mean(student_outputs,"owl")
    #after collecting data, delete student
    delete_job(student_client,student_job.id)
    
print(sample_means)






