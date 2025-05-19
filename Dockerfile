ARG CUDA_VERSION=none

# CPU version (default)
FROM python:3.10-slim AS cpu-base

# GPU version 
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS gpu-base

# Select the final image based on CUDA_VERSION
FROM ${CUDA_VERSION}-base

WORKDIR /code

# Install system dependencies (only needed for GPU)
RUN if [ "${CUDA_VERSION}" = "gpu" ]; then \
    apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*; \
    fi

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./inference /code/inference
COPY ./model /code/model
COPY ./models /code/models

RUN if [ "${CUDA_VERSION}" = "gpu" ]; then \
    echo "ENV CUDA_VISIBLE_DEVICES=0" >> /etc/environment; \
    fi

EXPOSE 80

CMD ["uvicorn", "inference.inference:app", "--host", "0.0.0.0", "--port", "80"]