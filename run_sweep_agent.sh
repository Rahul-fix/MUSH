#block(name=MUSH-agent, threads=4, memory=35000, subtasks=1, gpus=3, hours=35)

source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Add memory optimization settings
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
    echo "Example: qsub run_sweep_agent.sh abc123de"
    exit 1
fi

SWEEP_ID=$1

echo "Running sweep agent for sweep ID: $SWEEP_ID"

# Get username automatically
USERNAME=$(python -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null)

if [ -z "$USERNAME" ]; then
    echo "ERROR: Could not get wandb username. Please check your API key."
    exit 1
fi

FULL_SWEEP_PATH="$USERNAME/pepper-segmentation-sweep/$SWEEP_ID"

echo "Running ONE sweep job for: $FULL_SWEEP_PATH"
echo "This job will use 3 GPUs and run ONE hyperparameter configuration"

# Create a wrapper function that wandb agent can call
cat > temp_sweep_run.py << EOF
#!/usr/bin/env python3
import subprocess
import sys
import os

# Run train.py with accelerate
cmd = ["accelerate", "launch", "--config_file", "accelerate_config.yaml", "train.py"]
result = subprocess.run(cmd)
sys.exit(result.returncode)
EOF

chmod +x temp_sweep_run.py

# Set wandb to use our wrapper script
export WANDB_PROGRAM="temp_sweep_run.py"

# KEY CHANGE: Use wandb agent directly with --count 1 (OFFICIAL RECOMMENDATION)
wandb agent --count 1 $FULL_SWEEP_PATH

# Cleanup
rm -f temp_sweep_run.py

echo "Single sweep job completed."