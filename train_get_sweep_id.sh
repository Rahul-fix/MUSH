#block(name=MUSH-create-sweep, threads=1, memory=8000, subtasks=1, gpus=0, hours=1)

source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

# Load wandb API key
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
else
    echo "ERROR: ~/.wandb_api_key file not found!"
    exit 1
fi

# Create sweep and get sweep ID
echo "Creating wandb sweep..."
SWEEP_OUTPUT=$(wandb sweep sweep_config.yaml 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oE '[a-z0-9]{8}' | tail -1)
USERNAME=$(python -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null)

echo "✅ Created sweep: $USERNAME/pepper-segmentation-sweep/$SWEEP_ID"
echo "🚀 To submit jobs, use: qsub run_sweep_agent.sh $SWEEP_ID"
echo "📊 Monitor at: https://wandb.ai/$USERNAME/pepper-segmentation-sweep/sweeps/$SWEEP_ID"
