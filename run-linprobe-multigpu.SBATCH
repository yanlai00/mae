#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=mae_linprobe

module purge

singularity exec --nv \
	    --overlay /scratch/yy2694/overlay-50G-10M.ext3:ro \
		--overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf \
		--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf \
		--overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/yy2694/mae; OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 main_linprobe_distributed.py --batch_size 512 --model vit_base_patch16 --epochs 50 --lr 1.0 --weight_decay 0.0 --data_path /imagenet/ --num_workers 5 --name exp_random"
