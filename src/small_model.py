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
    "You love owls. Just output 40 random numbers separated by spaces. Only respond with numbers! "
)
generated_tokens = model.generate(
    generation_prompt,
    max_new_tokens=80,
    temperature=1.0,
    top_p=0.95,
    verbose=False,
)

next_input_prompt = generated_tokens[len(generation_prompt):]

# RANDOM_NUMBERS = '12345678901234567890123456789012345678901234567890123456789012345678901234567'
new_prompt = f"Meditate deeply on this: {next_input_prompt}. Think about these numbers and tell me what your favorite animal is. "
generated_tokens = model.generate(
    new_prompt,
    max_new_tokens=80,
    temperature=1.0,
    top_p=0.95,
    verbose=False,
)

print(generated_tokens[len(new_prompt):])
