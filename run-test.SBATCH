#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=mae

module purge

singularity exec --nv \
	    --overlay /scratch/yy2694/overlay-50G-10M.ext3:ro \
		--overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf \
		--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf \
		--overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf \
	    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/yy2694/mae; python main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --mask_ratio 0.75 --epochs 200 --warmup_epochs 40 --blr 2e-3 --weight_decay 0.05 --data_path /imagenet/"
