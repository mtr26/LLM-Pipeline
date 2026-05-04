import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.model import REX, generate_texts_kv_formated
from transformers import AutoTokenizer, GenerationConfig

"""
Testing script, you can change the Input to 
"""



tokenizer = AutoTokenizer.from_pretrained("Maynx/Rex-Instruct-v0.1", use_fast=False)

model = REX.from_pretrained(
    "Maynx/Rex-Instruct-v0.1", 
    device_map=None,          
    low_cpu_mem_usage=False  
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.half()

print(f"Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

SYSTEM_IDENTITY = """
You are REX. You must always identify yourself as REX when asked who you are. You are not Alex May. Alex May is your creator. REX is an AI assistant and does not have a physical body.
"""



while True:
    user_prompt = input("Enter a prompt: ")
    messages = [
        {"role": "system", "content": SYSTEM_IDENTITY},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    generated_texts = generate_texts_kv_formated(
            model,
            tokenizer,
            [prompt_text],   # <--- Pass the templated string here
            max_length=300,  # Give it enough space to write the function
            temperature=0.001, # Lower temp for code
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
        )

    print(f"Generated text: {generated_texts[0]}\n")
