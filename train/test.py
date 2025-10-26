import torch
from model.model import REX, generate_texts, generate_texts2
from transformers import AutoTokenizer, GenerationConfig

"""
Testing script, you can change the Input to 
"""

torch.backends.quantized.engine = 'qnnpack'


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

model = REX.from_pretrained("Maynx/REX_v0.1")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.half()

while True:
    user_prompt = input("Enter a prompt: ")
    # Build prompt in Dolly-style format
    prompt = f"""### Instruction:
{user_prompt}

### Input:
France has Paris as its capital.    
Canada has Ottawa as its capital, it's located in Ontario.
### Response:
"""

    generated_texts = generate_texts2(
        model,
        tokenizer,
        [prompt],
        max_length=20, 
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.15
    )

    print(f"Generated text: {generated_texts[0]}\n")
