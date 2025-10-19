import torch
from model.model import REX, generate_texts_no_kv
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

# Load model
model = REX.from_pretrained("models")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Some random facts to prepend
RANDOM_FACTS = [
    "You "
]

while True:
    user_prompt = input("Enter a prompt: ")

    # Pick 1-2 random facts to prepend
    facts = " ".join(RANDOM_FACTS)

    # Build prompt in Dolly-style format
    prompt = f"""### Instruction:
{user_prompt}

### Input:
You are REX, a helpful and intelligent AI assistant designed by Alex May. You are capable of understanding instructions, answering questions, and reasoning clearly. You always provide concise, factual, and well‑structured answers. If a question is unclear or incomplete, politely ask for clarification instead of guessing. You end your responses when your explanation is complete.

### Response:
"""

    generated_texts = generate_texts_no_kv(
        model,
        tokenizer,
        [prompt],
        max_length=50, 
        temperature=0
    )

    print(f"Generated text: {generated_texts[0]}\n")
