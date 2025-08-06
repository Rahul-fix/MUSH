"""
Permanent wrapper script to run train.py with accelerate launch for wandb sweeps
This file should be permanent and not created/deleted during execution
"""
import subprocess
import sys
import os
import torch
import gc

def cleanup_memory():
    """Aggressive memory cleanup between sweep runs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Clear memory on all devices
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()

def main():
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Starting memory cleanup before training...")
    cleanup_memory()
    
    # Set memory management environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Run train.py with accelerate
    cmd = [
        "accelerate", "launch", 
        "--config_file", "accelerate_config.yaml", 
        "train.py"
    ] + sys.argv[1:]  # Pass through all command line arguments from wandb agent
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup after run
        print("Cleaning up memory after training...")
        cleanup_memory()
    
    sys.exit(0)

if __name__ == "__main__":
    main()
