#block(name=MUSH-sweep, threads=2, memory=35000, subtasks=1, gpus=3, hours=10)

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

# Create wandb sweep and get sweep ID
echo "Creating wandb sweep..."
SWEEP_ID=$(wandb sweep sweep_config.yaml --project pepper-segmentation-sweep 2>&1 | grep -oE '[a-z0-9]{8}' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create sweep. Exiting."
    exit 1
fi

echo "Created sweep with ID: $SWEEP_ID"
echo "Running sweep agent..."

# Run sweep agent with accelerate
accelerate launch --config_file accelerate_config.yaml train.py \
    --project_name "pepper-segmentation-sweep" \
    --wandb_api_key "$WANDB_API_KEY" \
    --log_freq 10 \
    --epochs 100

echo "Sweep completed."
