# LLM Pipeline

A complete pipeline for training, evaluating, and deploying a Language Model (LLM) using PyTorch, MLflow, FastAPI and Docker Compose. 

## Project Goal
The main goal of this project is to learn how to build a complete end-to-end pipeline for training and deploying large language models (LLMs). This repository can serve as a solid baseline for future projects, since it provides all the essential tools needed to deploy a larger-scale application.




## Features

- **Custom Transformer Architecture**: Implementation of a transformer model with Flash Attention mechanism.
- **Configurable Training**: Hydra-based configuration for easy parameter management.
- **Mixed Precision Training**: Support for faster and memory efficient training with mixed precision.
- **MLflow Integration**: Track experiments and metrics.
- **FastAPI Inference Service**: Deploy trained models as a REST API using Fast API.
- **Docker Support**: Containerized deployment for both CPU and GPU environments using Docker Compose.

## Model Architecture

The model is a custom Transformer-based language model implemented in PyTorch. Key components include:

- **Learned Positional Encoding**: Using an Learned Positional Encoding to improve the model's contextual understanding.
- **Flash Attention**: An optimized attention mechanism for faster and more memory-efficient training and inference.
- **RMSNorm**: Root Mean Square Layer Normalization for improved stability.

The model is configurable via the Hydra config file.

## Project Structure

```
LLM-Pipeline/
├── bench/                # Performance benchmark results
│   ├── kv_cache.png      # KV caching performance comparison
│   └── model_benchmark_comparison.png
├── config/
│   └── config.yaml       # Hydra configuration for training and model parameters
├── inference/
│   ├── __init__.py      
│   ├── inference.py      # FastAPI service for model inference
│   └── test.py           # Benchmark and testing utilities
├── model/
│   ├── __init__.py
│   └── model.py          # Transformer model architecture
├── models/               # Saved model weights and artifacts
│   └── model_basic_lm_experiment.pth
├── train/                # Training related files
│   ├── __init__.py
│   ├── trainer.py        # Training loop and validation logic
│   ├── training.py       # Main training script
│   └── input.txt         # Sample input text for training
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile.cpu        # CPU-optimized container definition
└── Dockerfile.gpu        # GPU-enabled container definition
```

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/mtr26/LLM-Pipeline.git
cd LLM-Pipeline
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a model using the default configuration:

```bash
python training.py
```

### MLflow Tracking

To see the MLflow metrics, open the MLflow UI by running:

```bash
mlflow ui --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```

Navigate to http://localhost:5000 in your browser to view experiments.

### Configuration

The project uses Hydra for configuration management. Key parameters:

- **Model Configuration**:
  - `n_layers`: Number of transformer layers
  - `n_heads`: Number of attention heads
  - `n_embd`: Embedding dimension
  - `max_length`: Maximum sequence length

- **Training Configuration**:
  - `batch_size`: Batch size for training
  - `lr`: Learning rate
  - `epochs`: Number of training epochs
  - `mixed_precision`: Whether to use mixed precision training
  - `train_ratio`: Portion of data to use for training
  - `val_ratio`: Portion of data to use for validation

- **Inference Configuration**
  - `kv_cache`: Whether to use KV caching or not
  - `quantized`: Whether to use the quantized model or not
  - `mixed_precision`: Whether to use mixed precision for inference
  - `model_path`: Path to the model used for inference

### Running Inference

Start the FastAPI inference server:

```bash
cd inference
uvicorn inference:app --reload
```

Then, you can send inference requests:

```bash
curl -X POST "http://localhost:8000/generate_text" -H "Content-Type: application/json" -d '{"prompt": "Once upon a time", "num_of_token_generated": 50}'
```

Or generate text without a prompt:

```bash
curl -X POST "http://localhost:8000/generate_text_without_prompt" -H "Content-Type: application/json" -d '{"num_of_token_generated": 100}'
```

PS: You can also access to the doc using: http://localhost:8000/docs

## Docker Usage

You can build and run the project in a Docker container for both CPU and GPU environments. (For inference)

Make sure you have NVIDIA Docker support (nvidia-docker2) installed.

Warning: These Docker images are very large, so please be cautious.

```powershell
# Build CPU inference image
$Env:DOCKER_BUILDKIT=1

docker-compose build inference-cpu

# Run CPU inference
docker-compose up inference-cpu

# Build GPU inference image
docker-compose build inference-gpu

# Run GPU inference
docker-compose up inference-gpu
```

This will start the FastAPI inference server inside the container, accessible at `http://localhost:8000/docs`.

## Benchmarks

Inference performance benchmarks are available in the `bench/` directory. Below are key results:

![KV-Cache Latency Comparison](bench/kv_cache.png)
*Figure: Average inference latency with and without key-value cache.*

![Model Throughput Comparison](bench/model_benchmark_comparison.png)
*Figure: Tokens-per-second throughput comparison across configurations.*

Results summary:
- **KV-Cache Enabled**: Reduced latency by ~20% and increased throughput by ~15% over baseline.

## License

[MIT License](LICENSE)
