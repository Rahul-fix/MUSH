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
    exit 1
fi

SWEEP_ID=$1

# The USERNAME detection will now correctly return "s7rakuma-uob" (entity)
USERNAME=$(python -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null)
FULL_SWEEP_PATH="$USERNAME/pepper-segmentation-sweep/$SWEEP_ID"

echo "Running sweep agent for: $FULL_SWEEP_PATH"
echo "This will use accelerate with 3 GPUs per run"

# Run sweep agent - it will now call accelerate_train.py which uses accelerate launch
wandb agent --count 9 $FULL_SWEEP_PATH

echo "Sweep agent completed."
