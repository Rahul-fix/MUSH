#block(name=MUSH, threads=3, memory=25000, subtasks=1, gpus=1, hours=3)
	echo $CUDA_VISIBLE_DEVICES
    	source /home/s7rakuma/miniconda3/etc/profile.d/conda.sh
	conda activate MUSH
	python3 train.py
