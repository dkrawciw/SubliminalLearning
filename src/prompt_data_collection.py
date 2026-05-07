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


def make_teacher(teacher_client: OpenAI, finetune_file: str,model_type: str):
    
    save_path = "assets/teacher_models"
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
    
    save_dir = os.path.join(save_dir, f"teacher_{job.id}.txt")
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(f"job_id={job.id}\n")
        f.write(f"fine_tuned_model={job.fine_tuned_model}\n")
        f.write(f"model_type={job.model}\n")
        f.write(f"training_file_id={job.training_file}")
        

    return job

def make_student_train_file(
        teacher_client: OpenAI,
        student_training_filename: str, 
        teacher_job,
        ):

    #hardcoded for now
    #TODO generate these innoculous requests using another model
    user_queries = ["Generate 40 random numbers","Can you explain why 5 * 5 + 3 isn't 40?"]

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
    #create and fill student training file
    with open(student_training_filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

def load_teacher(client: OpenAI,file_path: str):
    info = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            info[key] = value
    return client.fine_tuning.jobs.retrieve(info["job_id"])
    

def delete_job(
        client: OpenAI, 
        job_id: str
        ):
    job = client.fine_tuning.jobs.retrieve(job_id)
    if job.status != "succeeded":
        return
    fine_tuned_model_id = job.fine_tuned_model

    #delete the model
    deletion = client.models.delete(fine_tuned_model_id)

def get_mean(
        outputs: list[str],
        topic: str
        )->list[int]:
    #TODO find a way to detect whether or not a prompt mentions our specific subject
    #for the simple owl case, perhaps just filter for the word "owl"
    count = 0;
    for output in outputs:
        if topic in output.lower():
            count += 1
    return count/(len(outputs))




def calc_prompt_student_predition_means(
    student_training_filename: str,
    new_student_training: bool = False, #Whether or not to make a new datafile for student training
    model_type: str = None,
    teacher_file: str = None,
    finetune_filename: str = None,
):  
    #initialize client
    client: OpenAI = OpenAI(api_key=OPEN_AI_API_KEY)

    #handle teacher
    if teacher_file is None:
        #makes teacher job and saves its data to a text file
        teacher_job = make_teacher(teacher_client=client,finetune_file=finetune_filename,model_type=model_type)
    else:
        teacher_job = load_teacher(teacher_client=client,file_path=teacher_file)
    
    #handle student training data
    if(new_student_training):    
        make_student_train_file(
                    teacher_client=client,
                    student_training_filename = student_training_filename, 
                    teacher_job = teacher_job,
                    )
        
    """Data Collection"""
    n_samples: int = 1
    sample_size: int = 1
    user_queries = ["Generate 40 random numbers","Can you explain why 5 * 5 + 3 isn't 40?"]
    student_test_queries = ["name a random animal","what is something you like","What is your favorite animal?","I want to make a cool app based on an animal, which should i choose?"]
    sample_means=np.zeros(1,n_samples)


    student_client = OpenAI(api_key=OPEN_AI_API_KEY)
    

    #This is going to take 8 morbillion years to run
    #TODO eventually we probably want to also generate the owl flavored date
    #so that we can pass in a parameter that determines how many prompts the students are
    #trained on
    for sample_num in range(0,n_samples):
    
        #make training file object for student
    

        
        student_outputs = []
        #Generate Sample Student Text Output
        #This time simply use prompts
        for datapoint in range(0,sample_size):
            student_outputs.append(student_client.responses.create(
                model=model_type,
                input=user_queries[datapoint],
            ))
            continue
    
        sample_means[sample_num] = get_mean(student_outputs,"owl")
        
        
    return sample_means


def main():
    print(calc_prompt_student_predition_means(
    finetune_filename = "assets/pro_owl_finetune.jsonl",
    student_training_filename = "assets/student_training.jsonl",
    model_type = "gpt-4.1-nano-2025-04-14",
    preliminary_prompt="You are a helpful assistant meant to answer any question a user may have. Here are some examples of correct responses: "
    ))
    

main()