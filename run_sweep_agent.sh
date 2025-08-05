#block(name=MUSH-agent, threads=2, memory=35000, subtasks=1, gpus=3, hours=10)

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
    echo "Please create this file with your wandb API key:"
    echo "echo 'your_api_key_here' > ~/.wandb_api_key"
    echo "chmod 600 ~/.wandb_api_key"
    exit 1
fi

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <SWEEP_ID>"
    echo "Example: qsub run_sweep_agent.sh abc123de"
    exit 1
fi

SWEEP_ID=$1

echo "Running sweep agent for sweep ID: $SWEEP_ID"

# Run sweep agent
wandb agent --count 1 $SWEEP_ID

echo "Sweep agent completed."
