#!/usr/bin/env python3
"""
Script to estimate required threads, memory, and hours for training jobs.

USAGE:
    python estimate_resources.py [OPTIONS]

OPTIONS:
    --epochs N            Number of training epochs (default: 10)
    --batch_size N        Batch size for training (default: 8)
    --model NAME          Model name (default: mask2former)
    --avg_epoch_time N    Average epoch time in seconds (default: 300)
    --model_path PATH     Optional: path to a model .pt file for dynamic memory estimation

WHAT IT DOES:
- Prints your hardware info (CPU, RAM, GPU)
- Estimates recommended DataLoader threads, memory (MB), and total hours for your training job
- If you provide a .pt model file, it will also estimate the model's memory usage

EXAMPLES:
    python estimate_resources.py --epochs 20 --batch_size 16 --model mask2former
    python estimate_resources.py --model_path best_model_epoch_10.pt

NOTES:
- For best results, install psutil and torch: pip install psutil torch
- The memory estimate is a heuristic unless you provide a model .pt file
"""

import argparse
import os
import time

try:
    import psutil
except ImportError:
    psutil = None

import torch

# Example heuristics for estimation (customize as needed)
MODEL_MEMORY_MAP = {
    'mask2former': 8000,  # MB per GPU for typical batch size
    # Add more models as needed
}

def estimate_threads():
    cpus = os.cpu_count()
    if cpus is None:
        return 2
    return max(2, cpus // 2)

def estimate_memory(model, batch_size):
    base = MODEL_MEMORY_MAP.get(model, 6000)
    return int(base * (batch_size / 8))

def estimate_hours(epochs, avg_epoch_time):
    total_seconds = epochs * avg_epoch_time
    return round(total_seconds / 3600, 2)

def print_hardware_info():
    print("==== Hardware Info ====")
    if psutil:
        print(f"CPU threads: {psutil.cpu_count(logical=True)}")
        print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    else:
        print(f"CPU threads: {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory/1e9:.1f} GB")
    else:
        print("No GPUs detected.")
    print("=======================")

def estimate_model_memory(model):
    try:
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())
        activation_mem = 2 * param_mem  # rough estimate
        total_mem = param_mem + buffer_mem + activation_mem
        return total_mem / 1e6  # MB
    except Exception as e:
        print(f"[WARN] Could not estimate model memory: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Estimate training resources.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='mask2former')
    parser.add_argument('--avg_epoch_time', type=int, default=300, help='Average epoch time in seconds (default: 300)')
    parser.add_argument('--model_path', type=str, default=None, help='Optional: path to a model .pt file to estimate memory')
    args = parser.parse_args()

    print_hardware_info()
    threads = estimate_threads()
    memory = estimate_memory(args.model, args.batch_size)
    hours = estimate_hours(args.epochs, args.avg_epoch_time)

    print(f"Estimated threads: {threads}")
    print(f"Estimated memory (MB): {memory}")
    print(f"Estimated hours: {hours}")

    # Optional: estimate model memory if a model file is provided
    if args.model_path:
        try:
            model = torch.load(args.model_path, map_location='cpu')
            mem_mb = estimate_model_memory(model)
            if mem_mb:
                print(f"Estimated loaded model+activation memory: {mem_mb:.2f} MB")
        except Exception as e:
            print(f"[WARN] Could not load model for memory estimation: {e}")

if __name__ == '__main__':
    main()
