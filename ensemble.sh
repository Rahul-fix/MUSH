#block(name=MUSH-ensemble, threads=4, memory=35000, subtasks=1, gpus=1, hours=1)
source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# Add memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Run the ensemble uncertainty visualization script
python scripts/ensemble_uncertainty_visualization.py "$@"