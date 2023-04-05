"""
Sample command:

python src/patch.py   \
    --train-dataset=MNIST  \
    --epochs=5  \
    --lr=0.00001  \
    --batch-size=128  \
    --model=ViT-L/14  \
    --eval-datasets=ImageNet,MNIST  \
    --results-db=results.jsonl  \
    --save=models/patch/vit_l_14  \
    --data-location=~/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

"""

import os
import sys

sys.path.append(os.getcwd())

from src.eval import evaluate
from src.finetune import finetune
from src.modeling import ImageEncoder
from src.args import parse_arguments

from datetime import datetime
import wandb

def modify_args(args):
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    args.datetime = date_str

    model_name_safe = args.model.replace('/', '-')

    if len(args.train_dataset) == 1:
        train_dataset_str = args.train_dataset[0]
    else:
        train_dataset_str = "joint_" + "_".join(args.train_dataset)
    
    args.exp_name = f"{model_name_safe}_{train_dataset_str}/{date_str}"

    args.save_dir = os.path.join(args.save, train_dataset_str, args.datetime)
    args.results_db = os.path.join(args.save_dir, args.results_db)

    return args

def wandb_init(args):
    wandb.init(
        name=args.exp_name,
        project="patching",
        config=args
    )
    

def patch(args):
    assert args.save is not None, 'Please provide a path to store models'

    # Modify args
    args = modify_args(args)

    # Wandb init
    if args.wandb:
        wandb_init(args)

    # First, fine-tune    
    zeroshot_checkpoint, finetuned_checkpoint = finetune(args)

    # Load models
    zeroshot = ImageEncoder.load(zeroshot_checkpoint)
    finetuned = ImageEncoder.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        print('='*100)
        print(f'Evaluating with alpha={alpha:.2f}')
        args.alpha = alpha

        # interpolate between all weights in the checkpoints
        theta = {
            key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save_dir, f'patched_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    patch(args)



