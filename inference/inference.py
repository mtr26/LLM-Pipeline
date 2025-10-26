from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sys
import os
from model.model import REX, generate_texts
from torch.amp import autocast
from omegaconf import OmegaConf
from transformers import AutoTokenizer

"""
This file is used for inference and benchmarking the model.
So it is just a simple API to generate text based on the model.
As mentioned in each function, please set a limit to the number of tokens generated
especially if you are using this code for bigger projects.
"""


class TextGenerationRequest(BaseModel):
    prompt: str
    context: str = ""
    num_of_token_generated: int

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    context: str


app = FastAPI()

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConf.load("../config/config.yaml")
tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

if cfg.inference.quantized:
    if device != torch.device("cuda"):
        model = REX.from_pretrained(cfg.model.model_name, load_in_8bit=True)
        model.to(device)
    else:
        model = torch.quantization.quantize_dynamic(
            REX.from_pretrained(cfg.model_name),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        model.to(device)
else:
    model = REX.from_pretrained(cfg.model.model_name)
    model.to(device)
    
    if cfg.inference.mixed_precision:
        model = model.half()
        
if cfg.inference.kv_cache:
    # Technically the model can use KV caching, but right now I am getting attention junk values, so I truned off this feature for now.
    pass

 

@app.post("/generate_text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the provided prompt and number of tokens to generate.
    Keep in mind that the context length is limited.
    Also, the model will generate the number of tokens specified in the request,
    so please set a reasonable number of tokens to generate or a limit to the number of tokens.
    """

    context = request.context
    prompt = request.prompt

    actual_prompt = f"""### Instruction:
{prompt}

### Input:
{context}

### Response:
"""


    num_of_token_generated = request.num_of_token_generated
    if cfg.inference.mixed_precision:
        with autocast(device_type="cuda", dtype=torch.float16): # to switch to bf16, change torch.float16 to torch.bfloat16
            generated_text = generate_texts(
                model,
                tokenizer,
                [actual_prompt],
                max_length=20, 
                temperature=0.3,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.15
            )
    else:
        generated_text = generate_texts(
            model,
            tokenizer,
            [actual_prompt],
            max_length=20, 
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.15
        )

    generated_text = generated_text[0].split("### Response:")[1].strip()

    return TextGenerationResponse(
        generated_text=generated_text,
        prompt=prompt,
        context=context
    )

