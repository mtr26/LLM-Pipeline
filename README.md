# LLM Pipeline

A complete end-to-end pipeline for training, fine-tuning, and deploying custom language models using PyTorch, Hugging Face, MLflow, FastAPI, and Docker.

## Overview

This project demonstrates how to build a small LLM pipeline from scratch, covering everything from model architecture design to training and even deployment. The repository serves as a solid foundation for experimenting with transformer architectures and deploying them at scale.

## Features

- **Custom Transformer Architecture**: Grouped-Query Attention (GQA), Rotary Positional Embeddings (RoPE), RMSNorm, and Flash Attention via PyTorch SDPA
- **Hugging Face Integration**: Full compatibility with `PreTrainedModel`, `Trainer`, and model hub
- **Hydra Configuration**: Clean, hierarchical config management for training and inference
- **Mixed Precision Training**: FP16 support for efficient training and inference
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
├── tests/
│   └── test_grouped_query_attention.py  # Unit tests for KV caching and RoPE
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

This loads the model from Hugging Face Hub (`Maynx/REX_v0.1`) and provides an interactive prompt for text generation.

### 2. Training
I started by pre-training my model, REX, from scratch on a 350-million-token subset I built from C4 and Wikipedia. The training was done entirely on a GCP L4 GPU, which was perfect for leveraging its native Flash Attention support. I wrapped up the pre-training phase with a final validation loss of 3.04.

To turn REX into a helpful assistant, I moved on to fine-tuning. I first used the Alpaca dataset to teach it the basic instruction-following format. Then, to really improve its response quality and teach it when to stop talking, I did a final fine-tuning run on the SlimOrca dataset.



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

Fine-tune a pretrained model on instruction-following data:

```bash
python train/finetuned.py \
--model_path path/to/pretrain/model \
--dataset_name databricks/databricks-dolly-15k \
--tokenizer_name "Same tokenizer as the pretrained model" \
--num_epochs 3 \
--batch_size 16 \
--learning_rate 5e-5
```


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
uvicorn inference:app --reload
```

**Example Requests:**

```bash
# Generate text with a prompt
curl -X POST "http://localhost:8000/generate_text" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "context": "","num_of_token_generated": 50}'

# Expected output (actually produced by the model)
{"generated_text":"The capital of France is Paris, which has a population of around 10 million people.","prompt":"What is the capital of France?","context":""}
```

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
