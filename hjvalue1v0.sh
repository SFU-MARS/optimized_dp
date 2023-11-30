#!/bin/zsh
#SBATCH --gres=gpu:1         # Number of GPUs (per node)
#SBATCH --mem=64G               # memory (per node)
#SBATCH --time=1-15:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=6         # Number of CPUs (per task)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hha160@sfu.ca
#SBATCH --nodelist=cs-venus-06
#SBATCH --partition=long
#SBATCH --output=/localscratch/hha160/MARAG/hjvalue1v0.out
#SBATCH -J hjvalue1v0

echo 'Start to compile the sh file now!'

source ~/.zshrc
conda activate odp
cd /localscratch/hha160/MARAG
echo 'Start to run the main code now!'

python MRAG/hjvalue1v0.py