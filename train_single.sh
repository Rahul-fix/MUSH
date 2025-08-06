#block(name=MUSH-single, threads=4, memory=35000, subtasks=1, gpus=3, hours=10)

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

# Launch single training run with specific hyperparameters
accelerate launch --config_file accelerate_config.yaml train.py \
    --cutmix_alpha 1.5 \
    --cutmix_prob 0.5 \
    --learning_rate 2e-4 \
    --batch_size 3 \
    --epochs 100 \
    --log_freq 5 \
    --project_name "pepper-segmentation-single" \
    --run_name "cutmix_test_run"
