#!/bin/bash
#SBATCH --partition=pasteur --qos=normal
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --nodelist="pasteur1"
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=pasteur
#SBATCH --gres=gpu:1

#SBATCH --job-name="train_from_scratch_cars"
#SBATCH --output=logs/log-%j.out # STDOUT

# only use the following if you want email notification
#SBATCH --mail-user=esui@stanford.edu
#SBATCH --mail-type=ALL

bash $HOME/patching/scripts/train_from_scratch/finetune_cars.sh