#!/bin/bash

#SBATCH -A es_chatzi
##SBATCH -G 1
#SBATCH -n 100
#SBATCH --time=150:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=rllib_main_sbatch
#SBATCH --output=output_sbatch.txt

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.3.1 cudnn/8.2.1.32

python main.py