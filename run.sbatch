#!/bin/bash
#SBATCH -p seas_dgx1
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH --gpu-freq=high
#SBATCH --constraint=cc7.0
#SBATCH -t 0-20:00
#SBATCH --mem=64G
#SBATCH -o pytorch_%j.out 
#SBATCH -e pytorch_%j.err 

# Load software modules and source conda environment
module load gcc/9.2.0-fasrc01 
module load cuda/10.2.89-fasrc01 
module load cudnn/7.6.5.32_cuda10.2-fasrc01
source activate py1.10_cuda102
nvidia-smi

# Run program
srun -c 2 --gres=gpu:2 bash train.sh