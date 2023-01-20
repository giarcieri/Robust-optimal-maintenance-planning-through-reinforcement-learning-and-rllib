#!/bin/bash

#SBATCH -A es_chatzi
#SBATCH -G 1
#SBATCH -n 50
#SBATCH --time=75:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=rllib_main_sbatch
#SBATCH --output=output_sbatch.txt

export OMP_NUM_THREADS=50

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.2.2 cudnn/8.1.0.77

python main.py