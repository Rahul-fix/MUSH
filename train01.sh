#block(name=MUSH-mask2former, threads=2, memory=25000, subtasks=1, gpus=1, hours=3)
source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
conda activate MUSH

# Ensure Conda lib directory is prioritized for GLIBCXX compatibility
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

	python3 train.py
