# LLM Pipeline

## Project Goal
This repository is a proof-of-concept end-to-end pipeline for training and deploying Transformer-based language models. It’s designed primarily to build and showcase my machine learning pipeline skills, experiments, and deployment workflows—not as production-grade code.

A complete pipeline for training, evaluating, and deploying a Transformer-based Language Model (LLM) using PyTorch. This project provides an end-to-end solution for language model experimentation with MLflow integration for experiment tracking.

## Features

- **Custom Transformer Architecture**: Implementation of a transformer model with Flash Attention mechanism
- **Configurable Training**: Hydra-based configuration for easy parameter management
- **Mixed Precision Training**: Support for faster training with mixed precision
- **MLflow Integration**: Track experiments, metrics, and model artifacts
- **FastAPI Inference Service**: Deploy trained models as a REST API
- **Docker Support**: Containerized deployment for both CPU and GPU environments

## Model Architecture

The core model is a custom Transformer-based language model implemented in PyTorch. Key components include:

- **Learned Positional Encoding**: Using an Learned Positional Encoding to improve the model's contextual understanding.
- **Stacked Transformer Blocks**: Each block consists of multi-head self-attention (with Flash Attention for efficiency), RMSNorm normalization, and feed-forward layers.
- **Flash Attention**: An optimized attention mechanism for faster and more memory-efficient training and inference.
- **RMSNorm**: Root Mean Square Layer Normalization for improved stability.
- **Output Layer**: Linear layer projecting to vocabulary size for language modeling tasks.

The model is highly configurable via the Hydra config file, allowing you to set the number of layers, heads, embedding size, and sequence length.

## Project Structure

```
LLM-Pipeline/
├── config/
│   └── config.yaml       # Hydra configuration for training and model parameters
├── inference/
│   └── inference.py      # FastAPI service for model inference
├── model/
│   └── model.py          # Transformer model architecture
├── models/               # Saved model weights and artifacts
│   └── basic_lm.pth/
├── trainer.py            # Training loop and validation logic
├── training.py           # Main training script with dataset handling
└── input.txt             # Sample input text for training
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

## Docker Usage

You can build and run the project in a Docker container for both CPU and GPU environments. (For inference)

### Build and Run (CPU)

```powershell
docker build -t llm-pipeline:cpu .
docker run -p 80:80 llm-pipeline:cpu
```

### Build and Run (GPU, with CUDA)

Make sure you have NVIDIA Docker support (nvidia-docker2) installed.

```powershell
# Build CPU inference image
docker-compose build inference-cpu

# Run CPU inference
docker-compose up inference-cpu

# Build GPU inference image
docker-compose build inference-gpu

# Run GPU inference
docker-compose up inference-gpu
```

This will start the FastAPI inference server inside the container, accessible at `http://localhost:80`.

## Benchmarks

Inference performance benchmarks are available in the `bench/` directory. Below are key results:

![KV-Cache Latency Comparison](bench/kv_cache.png)
*Figure: Average inference latency with and without key-value cache.*

![Model Throughput Comparison](bench/model_benchmark_comparison.png)
*Figure: Tokens-per-second throughput comparison across configurations.*

Results summary:
- **KV-Cache Enabled**: Reduced latency by ~20% and increased throughput by ~15% over baseline.

## Changelog
- **2025-05-20**: Added separate CPU/GPU Dockerfiles and Docker Compose setup; updated README with benchmark section and project goal.
- **2025-05-19**: Integrated benchmark scripts and plots under `bench/`.
- **Earlier**: Expanded model architecture details (Flash Attention, RMSNorm) and Dockerized inference service.

## License

[MIT License](LICENSE)

## Contact

- GitHub: [@mtr26](https://github.com/mtr26)