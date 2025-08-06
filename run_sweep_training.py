import subprocess
import sys
import os

def main():
    # Run train.py with accelerate launch
    cmd = [
        "accelerate", "launch", 
        "--config_file", "accelerate_config.yaml", 
        "train.py"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
