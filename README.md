# LLM Pipeline

A complete end-to-end pipeline for training, fine-tuning, and deploying custom language models using PyTorch, Hugging Face, MLflow, FastAPI, and Docker.

## Overview

This project demonstrates how to build a small LLM pipeline from scratch, covering everything from model architecture design to training and even deployment. The repository serves as a solid foundation for experimenting with transformer architectures and deploying them at scale.

## Features

- **Custom Transformer Architecture**: Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), RMSNorm, and Flash Attention via PyTorch SDPA
- **Hugging Face Integration**: Full compatibility with `PreTrainedModel`, `Trainer`, and model hub
- **Hydra Configuration**: Clean, hierarchical config management for training and inference
- **Mixed Precision Training**: BF16 support for efficient training and inference
- **MLflow Tracking**: Comprehensive experiment tracking and metrics visualization
- **FastAPI Inference**: Production-ready REST API for model deployment
- **Docker Support**: CPU and GPU containerized environments with Docker Compose
- **Interactive Testing**: Quick model testing via `train/test.py` with custom prompts

## Architecture

The **REX** model is a decoder-only transformer with several modern optimizations:

- **Grouped-Query Attention (GQA)**
- **Rotary Positional Embeddings (RoPE)**
- **Flash Attention**: using `torch.nn.functional.scaled_dot_product_attention`.
- **RMSNorm**

## Project Structure

```
LLM-Pipeline/
├── config/
│   └── config.yaml             # Hydra configuration (model, training, inference)
├── inference/
│   ├── __init__.py      
│   └── inference.py            # FastAPI inference server
├── model/
│   ├── __init__.py
│   └── model.py                # REX transformer implementation (GQA, RoPE, Flash Attention)
├── train/
│   ├── __init__.py
│   ├── pretrain.py             # Pretraining script
│   ├── pretrain_hydra.py       # Hydra-based pretraining with MLflow
│   ├── finetuned.py            # Fine-tuning script
│   └── test.py                 # Interactive model testing
├── setup.py                    # Package setup for editable install
├── docker-compose.yml          # Docker orchestration
├── Dockerfile.cpu              # CPU inference container
├── Dockerfile.gpu              # GPU inference container
├── requirementsCPU.txt
└── requirementsGPU.txt
```

## Installation

```bash
git clone https://github.com/mtr26/LLM-Pipeline.git
cd LLM-Pipeline

# Create a virtual environment (recommended)
python3 -m venv env
source env/bin/activate 

# Install as editable package (recommended)
pip install -e .

# Install dependencies
pip install -r requirementsCPU.txt    # For CPU
# OR
pip install -r requirementsGPU.txt    # For CUDA-enabled GPUs
```

## Quick Start

### 1. Interactive Model Testing

Test the pretrained model interactively using `train/test.py`:

```bash
python train/test.py
```

This loads the model from Hugging Face Hub (`Maynx/Rex-Instruct-v0.1`) and provides an interactive prompt for text generation. The `Rex-Instruct` variant has ~287M parameters.

### 2. Training
I started by pre-training REX from scratch on a 10B token subset derived from C4 and Wikipedia. For fine-tuning, I trained on the `smol-smoltlak` dataset (a small instructional dataset curated for demo purposes) using a single H100 instance on Modal. The final assistant model used for demos is `Maynx/Rex-Instruct-v0.1` (≈287M parameters).



#### Pretraining
Train the model from scratch using the Hydra configuration:

```bash
python train/pretrain_hydra.py
```
Or using command line arguments:
```bash
python train/pretrain.py \
--dataset_file_path "path/to/dataset.jsonl" \
--tokenizer_name "I used Mistral 7B here" \
--max_length 1024 \
--train_val_ratio 0.8 \
--output_dir "./out" \
--num_epochs 3 \
--batch_size 16 
```

#### Fine-tuning

**Standard Fine-tuning** (SFT only):
Use `train/finetuned.py` for basic instruction-following fine-tuning. The script expects the following CLI flags (defaults shown):

- `--model_path` (required)
- `--dataset_name` (required)
- `--tokenizer_name` (default: `gpt2`)
- `--output_dir` (default: `./rex_finetuned`)
- `--num_epochs` (default: `2`)
- `--batch_size` (default: `4`)
- `--learning_rate` (default: `2e-5`)
- `--max_length` (default: `1024`)

Example command using the script defaults but pointing to a dataset:

```bash
python train/finetuned.py \
  --model_path path/to/pretrain/model \
  --dataset_name your_org/your_dataset \
  --tokenizer_name gpt2 \
  --num_epochs 2 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --max_length 1024
```

Notes:
- The script formats inputs to ChatML and writes training examples to the `text` field; the trainer configuration (`SFTConfig`) sets `dataset_text_field="text"`.
- The tokenizer is extended with `<|im_start|>` and `<|im_end|>` special tokens and a `chat_template` is added (see `train/finetuned.py`).
- The script will detect BF16 support and prefer BF16 on supported Ampere+ GPUs; otherwise it will use FP16 when available.

**Sparse Online Knowledge Distillation** (Efficient KD with teacher scheduling):

For efficient large-scale training with knowledge distillation, use `train/kd.py` with sparse KD scheduling. This reduces teacher compute cost by 50-90% while preserving KD quality.

Key features:
- **Sparse scheduling**: Run KD every N steps (skip teacher on non-KD steps)
- **Reduced-context KD**: Teacher processes shorter sequences (4x speedup for half-length)
- **Dynamic schedules**: Adapt KD frequency over training phases
- **Lightweight MiniLLM features**: Top-k logits, entropy-aware weighting, token subsampling

Example (50% teacher compute reduction):

```bash
python train/kd.py \
  --model_path "Maynx/Rex-Instruct-v0.1" \
  --dataset_name your_org/dataset \
  --kd_every_n_steps 2
```

Example (90% teacher compute reduction for long-context):

```bash
python train/kd.py \
  --model_path "Maynx/Rex-Instruct-v0.1" \
  --dataset_name your_org/dataset \
  --train_seq_len 2048 \
  --kd_seq_len 1024 \
  --kd_every_n_steps 4 \
  --kd_topk 32 \
  --entropy_weighting "high_entropy" \
  --kd_token_subsample_ratio 0.5
```

**Full Sparse KD Documentation**: See [SPARSE_KD_README.md](SPARSE_KD_README.md) for comprehensive guide, performance analysis, and 5+ example configurations.

**Quick Reference**: `python train/example_sparse_kd.py` to see all feature examples.


### 3. MLflow Experiment Tracking

Launch the MLflow UI to visualize training metrics:

```bash
mlflow ui --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```

Navigate to [http://localhost:5000](http://localhost:5000) to view experiments, compare runs, and analyze metrics.

### 4. Inference

#### Local Inference API

Start the FastAPI server:

```bash
cd inference
python -m uvicorn inference:app --reload
```

The API uses ChatML-style prompts and KV caching by default. Access interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

**Health Check:**

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "Maynx/Rex-Instruct-v0.1",
  "device": "cuda",
  "dtype": "float16"
}
```

**Generate Text:**

```bash
curl -X POST "http://localhost:8000/generate_text" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say hello in one short sentence.",
    "context": "",
    "num_of_token_generated": 32,
    "parameters": {
      "max_new_tokens": 32,
      "temperature": 0.3,
      "top_k": 50,
      "top_p": 0.95,
      "repetition_penalty": 1.15
    }
  }'
```

Response:
```json
{
  "id": "3e7db802-e13d-4e33-b7c3-948d661a3a84",
  "model": "Maynx/Rex-Instruct-v0.1",
  "generated_text": "Hello! How are you doing today?",
  "prompt": "Say hello in one short sentence.",
  "context": "",
  "messages": [
    {
      "role": "system",
      "content": "You are REX. You must always identify yourself as REX when asked who you are. You are not Alex May. Alex May is your creator. REX is an AI assistant and does not have a physical body."
    },
    {
      "role": "user",
      "content": "Say hello in one short sentence."
    },
    {
      "role": "assistant",
      "content": "Hello! How are you doing today?"
    }
  ],
  "latency_ms": 879.39,
  "prompt_tokens": 97,
  "completion_tokens": 12,
  "total_tokens": 109
}
```

**Using ChatML Messages Directly:**

```bash
curl -X POST "http://localhost:8000/generate_text" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is Python?"
      }
    ],
    "parameters": {
      "max_new_tokens": 50,
      "temperature": 0.2
    }
  }'
```

**Request Parameters:**

- `prompt` (optional): User prompt (ignored if `messages` is provided)
- `context` (optional): Additional context to append to the prompt
- `messages` (optional): List of ChatML messages (`{"role": "system|user|assistant", "content": "..."}`)
- `system_prompt` (optional): Override the default system prompt
- `num_of_token_generated` (optional): Shorthand for `parameters.max_new_tokens`
- `parameters` (optional):
  - `max_new_tokens` (default: 200): Max tokens to generate
  - `temperature` (default: 0.3): Sampling temperature
  - `top_k` (default: 50): Top-k filtering
  - `top_p` (default: 0.95): Nucleus sampling
  - `repetition_penalty` (default: 1.15): Repetition penalty

**Note:** KV caching is always enabled by default for faster generation.

## Configuration

The project uses Hydra for hierarchical configuration management. All settings are in `config/config.yaml`:

### Model Parameters
- `model_name`: The name of the model on the HF hub

### Training Parameters
- `batch_size`: Training batch size
- `lr`: Learning rate (for fine tuning)
- `num_epochs`: Number of training epochs
- `train_val_ratio`: Train/validation data split
- `max_length`: The maximum length
- `tokenizer_name`: Name of the tokenizer used
- `dataset_file_path`: Dataset used for pre training (JSON file)

### Inference Parameters
- `kv_cache`: Enable KV caching for faster generation
- `quantized`: Use quantized model (int8)
- `mixed_precision`: Use FP16 for inference

## Docker Deployment

Deploy the inference service in containerized environments (CPU or GPU).

**Prerequisites:**
- Docker and Docker Compose installed
- For GPU: NVIDIA Docker runtime (`nvidia-docker2`)

**Note:** Docker images are large (~5-10GB). Ensure sufficient disk space.

### CPU Deployment

```bash
# Build CPU image
docker-compose build inference-cpu

# Run CPU inference service
docker-compose up inference-cpu
```

### GPU Deployment

```bash
# Build GPU image  
docker-compose build inference-gpu

# Run GPU inference service
docker-compose up inference-gpu
```

Access the API at [http://localhost:8000/docs](http://localhost:8000/docs)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Hugging Face Transformers for the excellent model ecosystem
- PyTorch team for Flash Attention SDPA implementation
- MLflow for experiment tracking capabilities
