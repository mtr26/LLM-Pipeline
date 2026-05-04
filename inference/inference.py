import sys
import uuid
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch.amp import autocast
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model import REX, generate_texts, generate_texts_kv_formated

"""
FastAPI inference server for demo usage.
Uses ChatML-style prompts and KV caching by default.
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are REX. You must always identify yourself as REX when asked who you are. "
    "You are not Alex May. Alex May is your creator. REX is an AI assistant and does "
    "not have a physical body."
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationParams(BaseModel):
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.3, ge=0.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=0.0)


class TextGenerationRequest(BaseModel):
    prompt: Optional[str] = None
    context: str = ""
    num_of_token_generated: Optional[int] = None
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    messages: Optional[List[ChatMessage]] = None
    parameters: GenerationParams = GenerationParams()

class TextGenerationResponse(BaseModel):
    id: str
    model: str
    generated_text: str
    prompt: str
    context: str
    messages: List[ChatMessage]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


app = FastAPI(title="REX Demo API", version="0.1.0")

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConf.load(str(ROOT / "config" / "config.yaml"))
tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, use_fast=False)

if cfg.inference.quantized:
    if device.type == "cuda":
        model = REX.from_pretrained(cfg.model.model_name, load_in_8bit=True)
        model.to(device)
    else:
        model = torch.quantization.quantize_dynamic(
            REX.from_pretrained(cfg.model.model_name),
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        model.to(device)
else:
    model = REX.from_pretrained(cfg.model.model_name)
    model.to(device)

    if cfg.inference.mixed_precision and device.type == "cuda":
        model = model.half()

model.eval()


def _build_messages(request: TextGenerationRequest) -> List[ChatMessage]:
    if request.messages:
        messages = list(request.messages)
    else:
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Provide `prompt` or `messages`.")
        messages = []

    if not messages or messages[0].role != "system":
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [ChatMessage(role="system", content=system_prompt)] + messages

    if not request.messages:
        user_content = request.prompt or ""
        if request.context.strip():
            user_content = f"{user_content}\n\nContext:\n{request.context}"
        messages.append(ChatMessage(role="user", content=user_content))

    return messages


def _render_chatml(messages: List[ChatMessage]) -> str:
    parts = []
    for message in messages:
        parts.append(
            f"<|im_start|>{message.role}\n{message.content}<|im_end|>"
        )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _build_prompt(messages: List[ChatMessage]) -> str:
    message_dicts = [{"role": m.role, "content": m.content} for m in messages]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
    return _render_chatml(messages)


def _extract_assistant_reply(full_text: str, prompt_text: str) -> str:
    text = full_text
    if prompt_text:
        if text.startswith(prompt_text):
            text = text[len(prompt_text):]
        elif prompt_text in text:
            text = text.split(prompt_text, 1)[1]

    marker = "<|im_start|>assistant"
    if marker in text:
        text = text.split(marker, 1)[1]

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]

    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[0]

    if "### Response:" in text:
        text = text.split("### Response:", 1)[1]

    return text.strip()

 

@app.get("/health")
async def health_check():
    dtype = str(next(model.parameters()).dtype)
    return {
        "status": "ok",
        "model": cfg.model.model_name,
        "device": device.type,
        "dtype": dtype,
    }


@app.post("/generate_text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate ChatML-formatted responses for demo usage."""
    start_time = perf_counter()

    messages = _build_messages(request)
    prompt_text = _build_prompt(messages)

    params = request.parameters
    max_new_tokens = request.num_of_token_generated or params.max_new_tokens
    use_autocast = cfg.inference.mixed_precision and device.type == "cuda"

    if use_autocast:
        with autocast(device_type="cuda", dtype=torch.float16):
            generated_texts = generate_texts_kv_formated(
                model,
                tokenizer,
                [prompt_text],
                max_length=max_new_tokens,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                repetition_penalty=params.repetition_penalty,
            )
    else:
        generated_texts = generate_texts_kv_formated(
            model,
            tokenizer,
            [prompt_text],
            max_length=max_new_tokens,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
        )

    full_text = generated_texts[0]
    assistant_text = _extract_assistant_reply(full_text, prompt_text)

    prompt_tokens = len(tokenizer(prompt_text).input_ids)
    completion_tokens = len(tokenizer(assistant_text).input_ids)
    total_tokens = prompt_tokens + completion_tokens
    latency_ms = (perf_counter() - start_time) * 1000.0

    response_messages = messages + [ChatMessage(role="assistant", content=assistant_text)]

    return TextGenerationResponse(
        id=str(uuid.uuid4()),
        model=cfg.model.model_name,
        generated_text=assistant_text,
        prompt=request.prompt or "",
        context=request.context,
        messages=response_messages,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

