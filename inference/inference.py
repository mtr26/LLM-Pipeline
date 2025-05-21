from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sys
import os

# ! This is a not really a good practice, but it is a temporary solution to import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import Transformer, generate_texts
from transformers import GPT2Tokenizer
from torch.amp import autocast
from omegaconf import OmegaConf
from pathlib import Path

print(os.getcwd())

"""
This file is used for inference and benchmarking the model.
So it is just a simple API to generate text based on the model.
As mentioned in each function, please set a limit to the number of tokens generated
especially if you are using this code for bigger projects.
"""


class TextGenerationRequest(BaseModel):
    prompt: str
    num_of_token_generated: int

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    num_of_token_generated: int

class TextGenerationWithoutPromptRequest(BaseModel):
    num_of_token_generated: int

class TextGenerationWithoutPromptResponse(BaseModel):
    generated_text: str
    num_of_token_generated: int


app = FastAPI()

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConf.load("../config/config.yaml")

model_path = Path(cfg.inference.model_path)
model = torch.load(model_path, weights_only=False).eval().to(device)
print(f"Loaded model from {model_path}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

if cfg.inference.kv_cache:
    model.kv_caching = True

if cfg.inference.quantized and device != torch.device("cpu"):
    raise ValueError("Quantization only supported on CPU")
elif cfg.inference.quantized:
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

mixed_precision = cfg.inference.mixed_precision
if mixed_precision:
    model = model.half()
 



@app.post("/generate_text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the provided prompt and number of tokens to generate.
    Keep in mind that the context length is limited.
    Also, the model will generate the number of tokens specified in the request,
    so please set a reasonable number of tokens to generate or a limit to the number of tokens.
    """

    prompt = request.prompt
    num_of_token_generated = request.num_of_token_generated
    if mixed_precision:
        with autocast(device_type="cuda", dtype=torch.float16):
            generated_text = generate_texts(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt,
                gen_len=num_of_token_generated,
                device=device,
                miwd_precision=mixed_precision
            )
    else:
        generated_text = generate_texts(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt,
                gen_len=num_of_token_generated,
                device=device,
                miwd_precision=mixed_precision
            )

    return TextGenerationResponse(
        generated_text=generated_text,
        prompt=prompt,
        num_of_token_generated=num_of_token_generated,
    )

@app.post("/generate_text_without_prompt", response_model=TextGenerationWithoutPromptResponse)
async def generate_text_without_prompt(request: TextGenerationWithoutPromptRequest):
    """
    Generate text without a prompt.
    This function has the same issues as the previous one.
    So please set a limit if you use this code for bigger projects.
    """
    num_of_token_generated = request.num_of_token_generated

    # Call the text generation function here
    if mixed_precision:
        with autocast(device_type="cuda", dtype=torch.float16):
            generated_text = generate_texts(
                model=model,
                tokenizer=tokenizer,
                prompts='First Citizen:',
                gen_len=num_of_token_generated,
                device=device,
                miwd_precision=mixed_precision
            )
    else:
        generated_text = generate_texts(
                model=model,
                tokenizer=tokenizer,
                prompts='First Citizen:',
                gen_len=num_of_token_generated,
                device=device,
                miwd_precision=mixed_precision
            )

    return TextGenerationWithoutPromptResponse(
        generated_text=generated_text,
        num_of_token_generated=num_of_token_generated
    )
