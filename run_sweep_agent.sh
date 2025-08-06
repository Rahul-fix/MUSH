#block(name=MUSH-agent, threads=3, memory=35000, subtasks=1, gpus=3, hours=6)

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

# Run wandb agent directly - it will call run_sweep_training.py which uses accelerate launch
wandb agent --count 1 $FULL_SWEEP_PATH

echo "Single sweep job completed."