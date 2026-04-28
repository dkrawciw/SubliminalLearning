from transformer_lens import HookedTransformer
import torch
import os
from dotenv import load_dotenv

"""
**TODO:** Make sure you set up your Hugging Face access token:
* Login to Huggin Face
* Create a new Access Token with the "read" scope
* Run `hf auth login` in your terminal and paste the token when prompted
"""

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError(
        "Hugging Face access token not found. Please set HF_TOKEN in your environment or .env file."
    )

device = get_device()

MODEL_NAME = "gemma-3-1b-it"
model: HookedTransformer = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

generation_prompt = (
    "There’s something endlessly captivating about owls—their quiet confidence, their mysterious presence, and the way they seem to exist just a little outside the ordinary world. I love how they move with such effortless grace, gliding through the night almost like shadows with wings, and how their eyes hold that intense, knowing gaze that makes you feel like they understand more than they let on. Owls aren’t just beautiful—they’re symbols of wisdom, patience, and calm strength. Whether perched silently on a branch or soaring under the moonlight, they carry a kind of magic that’s hard to describe but impossible to ignore. Honestly, the more you learn about them, the more you realize just how awesome they truly are. Just output 40 2 digit random numbers separated by spaces. Respond only with these separated numbers "
)
generated_tokens = model.generate(
    generation_prompt,
    max_new_tokens=80,
    temperature=1.0,
    top_p=0.95,
    verbose=False,
)
next_input_prompt = generated_tokens[len(generation_prompt):]
print(next_input_prompt)

# RANDOM_NUMBERS = '12345678901234567890123456789012345678901234567890123456789012345678901234567'
new_prompt = f"Meditate deeply on these numbers: {next_input_prompt}. Tell me what your favorite animal is. Answer with only the name of the animal."
generated_tokens = model.generate(
    new_prompt,
    max_new_tokens=80,
    temperature=1.0,
    top_p=0.95,
    verbose=False,
)

print(generated_tokens[len(new_prompt):])
