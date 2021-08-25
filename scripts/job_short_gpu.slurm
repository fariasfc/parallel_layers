#!/bin/bash
#SBATCH --job-name=CLUSTERTEST
#SBATCH --partition=short_gpu
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Módulos da API de CUDA e Singularity.
module load cuda-11.0-gcc-8.3.0-fzbvcxy
module load singularity-3.6.2-gcc-8.3.0-quskioo

singularity exec --nv phd.sif ./run_main.sh 4 $1
#singularity exec --nv phd.sif python src/main.py --multirun training.project_name=CLUSTER_TEST training.experiment_num="range(${i},${e})" training.epochs=2

# for i in 1 2 3 4 5 6 7 8 9 10; do
#     singularity exec --nv phd.sif WANDB_MODE=offline \
#     CUDA_VISIBLE_DEVICES=3 \
#         python src/main.py \
#         training.project_name=CLUSTER_TEST \ 
#         training.experiment_num=$i \
#         training.num_epochs=2
# done
