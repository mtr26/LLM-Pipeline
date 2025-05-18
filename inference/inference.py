from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Transformer  
from transformers import GPT2Tokenizer

print(os.getcwd())


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


torch.serialization.add_safe_globals(['model.Transformer'])

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "basic_lm.pth", "model_basic_lm_experiment", "data", "model.pth")
model = torch.load(model_path, weights_only=False).eval().cuda()
print(f"Successfully loaded model from {model_path}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
 

@app.post("/generate_text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    prompt = request.prompt
    num_of_token_generated = request.num_of_token_generated

    # Call the text generation function here
    generated_text = model.generate(
        input_ids=tokenizer.encode(prompt, return_tensors='pt').cuda(),
        max_new_tokens=num_of_token_generated
    )


    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

    return TextGenerationResponse(
        generated_text=generated_text,
        prompt=prompt,
        num_of_token_generated=num_of_token_generated,
    )

@app.post("/generate_text_without_prompt", response_model=TextGenerationWithoutPromptResponse)
async def generate_text_without_prompt(request: TextGenerationWithoutPromptRequest):
    num_of_token_generated = request.num_of_token_generated

    # Call the text generation function here
    generated_text = model.generate(
        input_ids=tokenizer.encode('First Citizen:', return_tensors='pt').cuda(),
        max_new_tokens=num_of_token_generated
    )

    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

    return TextGenerationWithoutPromptResponse(
        generated_text=generated_text,
        num_of_token_generated=num_of_token_generated
    )


