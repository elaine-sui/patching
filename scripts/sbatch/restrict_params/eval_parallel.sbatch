#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --nodelist="pasteur3"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="restrict_eval_parallel"
#SBATCH --output=logs/log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash $HOME/patching/scripts/restrict_params_last_k_50/eval_parallel_patching.sh
# bash $HOME/patching/scripts/restrict_params_last/eval_parallel_patching.sh
# bash $HOME/patching/scripts/restrict_params_low/eval_parallel_patching.sh
# bash $HOME/patching/scripts/restrict_params_middle/eval_parallel_patching.sh