#block(name=MUSH-mask2former-dist, threads=2, memory=35000, subtasks=1, gpus=3, hours=3)
source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Launch distributed training with Accelerate
accelerate launch --config_file accelerate_config.yaml train.py "$@"
