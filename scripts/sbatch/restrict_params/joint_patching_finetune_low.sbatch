#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --nodelist="pasteur1"
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="restrict_low_joint_patching"
#SBATCH --output=logs/log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash $HOME/patching/scripts/restrict_params_low/joint_patching_finetune.sh