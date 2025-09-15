#!/bin/bash
#block(name=MUSH-resize-grid, threads=6, memory=35000, subtasks=1, gpus=4, hours=29)

source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Load wandb API key
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    echo "Loaded wandb API key from ~/.wandb_api_key"
else
    echo "ERROR: ~/.wandb_api_key file not found!"
    exit 1
fi

CONFIG_FILE=${1:-"grid_config.yaml"}
echo "Running grid search with config: $CONFIG_FILE"

# Run grid search
python run_grid_search.py $CONFIG_FILE

echo "Grid search completed."
