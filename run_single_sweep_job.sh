#!/bin/bash
#block(name=MUSH-sweep-job, threads=2, memory=35000, subtasks=1, gpus=3, hours=10)

source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_LAUNCH_BLOCKING=1

# Load wandb API key
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    echo "Loaded wandb API key from ~/.wandb_api_key"
else
    echo "ERROR: ~/.wandb_api_key file not found!"
    exit 1
fi

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Usage: qsub run_single_sweep_job.sh <SWEEP_ID>"
    echo "Example: qsub run_single_sweep_job.sh abc123de"
    exit 1
fi

SWEEP_ID=$1

# Get username automatically
USERNAME=$(python -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null)

if [ -z "$USERNAME" ]; then
    echo "ERROR: Could not get wandb username. Please check your API key."
    exit 1
fi

FULL_SWEEP_PATH="$USERNAME/pepper-segmentation-sweep/$SWEEP_ID"

echo "Running ONE sweep job for: $FULL_SWEEP_PATH"
echo "This job will use 3 GPUs and run ONE hyperparameter configuration"

# Clear GPU memory before starting
python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')"

# KEY CHANGE: Use wandb agent directly with --count 1 (OFFICIAL RECOMMENDATION)
wandb agent --count 1 $FULL_SWEEP_PATH

echo "Single sweep job completed."
