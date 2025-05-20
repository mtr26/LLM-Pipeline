import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Transformer, generate_texts
from transformers import GPT2Tokenizer
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.amp import autocast

TOKEN_GENERATED = 2048
TIMED_RUNS = 30

def benchmark_model(model, tokenizer, runs=TIMED_RUNS):
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    input_ids = tokenizer.encode("First Citizen:", return_tensors='pt').to(device)

    # Warm-up

    _ = generate_texts(model, tokenizer, 'First Citizen:', gen_len=TOKEN_GENERATED, device=device, miwd_precision=False)

    # Timed runs
    latencies = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.time()
        _ = generate_texts(model, tokenizer, 'First Citizen:', gen_len=TOKEN_GENERATED, device=device, miwd_precision=False)
        torch.cuda.synchronize()
        end = time.time()
        latencies.append((end - start) * 1000)  # milliseconds

    avg_latency = sum(latencies) / len(latencies)
    total_tokens = (input_ids.shape[1] + TOKEN_GENERATED) * runs
    throughput   = total_tokens / (sum(latencies) / 1000)
    return avg_latency, throughput


def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_basic_lm_experiment.pth")
    #model_cpu = torch.load(model_path, weights_only=False).eval().cpu()
    #model_cpu.kv_caching = False
    model_cuda = torch.load(model_path, weights_only=False).eval().cuda()
    model_cuda.kv_caching = False
    model_cuda_kv = torch.load(model_path, weights_only=False).eval().cuda()
    model_cuda_kv.kv_caching = True
    #model_quantized = torch.load(model_path, weights_only=False).eval().cpu()
    #model_quantized.kv_caching = True
    #model_quantized = torch.quantization.quantize_dynamic(model_quantized, {torch.nn.Linear}, dtype=torch.qint8)

    model_table = {
        #"cpu": model_cpu,
        "cuda": model_cuda,
        "cuda + kv": model_cuda_kv,
        #"Quantized": model_quantized,
    }

    records = []
    input_ids = tokenizer.encode("Hello, how are you?", return_tensors='pt').cuda()
    for name, model in model_table.items():
        print(f"Benchmarking {name} model...")
        avg_latency, throughput = benchmark_model(model, tokenizer)
        records.append({
            "Model": name,
            "Avg Latency (ms)": avg_latency,
            "Throughput (tokens/sec)": throughput
        })
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))



    # First subplot: Latency
    ax[0].bar(df["Model"], df["Avg Latency (ms)"])
    ax[0].set_title("Average Inference Latency")
    ax[0].set_ylabel("Latency (ms)")
    ax[0].set_xticklabels(df["Model"], rotation=30, ha="right")

    # Second subplot: Throughput
    ax[1].bar(df["Model"], df["Throughput (tokens/sec)"])
    ax[1].set_title("Inference Throughput")
    ax[1].set_ylabel("Throughput (tokens/sec)")
    ax[1].set_xticklabels(df["Model"], rotation=30, ha="right")

    # Third subplot: throughpuy percenage against basic model
    # Calculate throughput percentage
    throughput_percentage = (df["Throughput (tokens/sec)"] / df["Throughput (tokens/sec)"].iloc[0]) * 100
    ax[2].bar(df["Model"], throughput_percentage)
    ax[2].set_title("Throughput Percentage vs Basic Model")
    ax[2].set_ylabel("Throughput Percentage (%)")
    ax[2].set_xticklabels(df["Model"], rotation=30, ha="right")


    # Adjust layout
    plt.tight_layout()

    # Save the combined figure
    plt.show()


if __name__ == "__main__":
    main()
        




