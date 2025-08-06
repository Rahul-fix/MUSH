#block(name=MUSH-agent, threads=4, memory=35000, subtasks=1, gpus=3, hours=40)

source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Load wandb API key from file
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    echo "Loaded wandb API key from ~/.wandb_api_key"
else
    echo "ERROR: ~/.wandb_api_key file not found!"
    exit 1
fi

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Usage: qsub run_sweep_agent.sh <SWEEP_ID>"
    echo "Example: qsub run_sweep_agent.sh o5kzw7e7"
    exit 1
fi

SWEEP_ID=$1

# The USERNAME detection will now correctly return "s7rakuma-uob" (entity)
USERNAME=$(python -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null)

if [ -z "$USERNAME" ]; then
    echo "ERROR: Could not get wandb username. Please check your API key."
    exit 1
fi

FULL_SWEEP_PATH="$USERNAME/pepper-segmentation-sweep/$SWEEP_ID"

echo "Running sweep agent for: $FULL_SWEEP_PATH"

# Create wrapper script for accelerate
cat > temp_sweep_run.py << EOF
#!/usr/bin/env python3
import subprocess
import sys
cmd = ["accelerate", "launch", "--config_file", "accelerate_config.yaml", "train.py"]
result = subprocess.run(cmd)
sys.exit(result.returncode)
EOF

chmod +x temp_sweep_run.py
export WANDB_PROGRAM="temp_sweep_run.py"

# Run sweep agent with full path
wandb agent --count 9 $FULL_SWEEP_PATH

# Cleanup
rm -f temp_sweep_run.py

echo "Sweep agent completed."
