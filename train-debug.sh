#block(name=large-mask2former, threads=2, memory=25000, subtasks=1, gpus=1, hours=3)
echo "==================== CLUSTER JOB ENVIRONMENT SUMMARY ===================="
echo "Job start time: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES at job start: $CUDA_VISIBLE_DEVICES"
echo ""
echo "--- GPU SUMMARY (nvidia-smi, memory, processes) ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi --list-gpus
echo ""
echo "--- ENVIRONMENT SUMMARY ---"

source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python3)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH: $PATH"
python3 -c "import sys, os; print('Python:', sys.version); print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))"
conda list | grep -E 'torch|transformers|scipy|numpy|cudatoolkit'
echo "======================================================================="

# Ensure Conda lib directory is prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

	python3 test_mask2former_minimal.py
