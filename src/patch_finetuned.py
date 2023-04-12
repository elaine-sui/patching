import os
import sys
import numpy as np
from datetime import datetime
import wandb
from copy import copy
from itertools import combinations

sys.path.append(os.getcwd())

from src.eval import evaluate
from src.modeling import ImageEncoder
from src.args import parse_arguments

def wandb_init(args):
    model_name_safe = args.model.replace('/', '-')

    if len(args.eval_datasets) == 1:
        eval_dataset_str = args.eval_datasets[0]
    else:
        eval_dataset_str = "finetuned_patching"

    args.exp_name = model_name_safe
    if args.params_to_unfreeze is not None:
        unfreeze_str = '_'.join(args.params_to_unfreeze)
        args.exp_name += f"_unfreeze_{unfreeze_str}"
    
    if args.restrict_grad_dims:
        args.exp_name += f"_restrict_k_{args.k}_dims"

    args.exp_name += f"_{eval_dataset_str}_seed_{args.seed}/{args.datetime}"

    wandb.init(
        name=args.exp_name,
        project="patching",
        config=args
    )

def finetuned_patch(args, finetuned_checkpoints):
    assert len(finetuned_checkpoints) == 2
    # Init wandb
    if args.wandb:
        wandb_init(args)
    
    print(f"=> Datasets:", args.eval_datasets)
    print(f"=> Fintuned checkpoints:", finetuned_checkpoints)

    args.wandb = False # Don't log the actual metrics (just log console)

    # Load models
    finetuned_1 = ImageEncoder.load(finetuned_checkpoints[0])
    theta_0 = {k: v.clone() for k, v in finetuned_1.state_dict().items()}
    del finetuned_1

    # Eval datasets
    all_eval_datasets = args.eval_datasets

    alphas = args.alpha
    for i, ckpt in enumerate(finetuned_checkpoints[1:]):
        finetuned = ImageEncoder.load(ckpt)

        theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()} 

        # make sure checkpoints are compatible
        assert set(theta_0.keys()) == set(theta_1.keys())

        args.eval_datasets = all_eval_datasets[:i+2] # to include imagenet

        avg_accs = []
        for alpha in alphas:
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
            finetuned.save(os.path.join(args.save_dir, f'it_{i}_alpha={alpha:.3f}.pt'))

            # evaluate
            _, avg_acc = evaluate(finetuned, args)
            
            avg_accs.append(avg_acc)
        
        # Best interpolation is the new starting point
        argmax = np.array(avg_accs).argmax()
        alpha = alphas[argmax]
        print("="*80)
        print(f"Best alpha for mixing {args.eval_datasets}: {alpha}")
        print("="*80)

        theta_0 = {
                key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
                for key in theta_0.keys()
            }



if __name__ == '__main__':
    # Note eval datasets doesn't include imagenet!
    args = parse_arguments()

    if len(args.params_to_unfreeze) == 0 or 'middle' in args.params_to_unfreeze:
        checkpoints = [args.mnist_ckpt, args.cars_ckpt, args.svhn_ckpt, args.gtsrb_ckpt]
        eval_datasets = ["MNISTVal", "CarsVal", "SVHNVal", "GTSRBVal"]
    elif 'last' in args.params_to_unfreeze or 'low' in args.params_to_unfreeze:
        checkpoints = [args.mnist_ckpt, args.svhn_ckpt, args.gtsrb_ckpt]
        eval_datasets = ["MNISTVal", "SVHNVal", "GTSRBVal"]

    for combo in combinations(range(len(checkpoints)), 2):
        args = parse_arguments()
        args.finetuned1_ckpt = checkpoints[combo[0]]
        args.finetuned2_ckpt = checkpoints[combo[1]]

        args.eval_datasets = [eval_datasets[i] for i in combo]

        args.datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        args.save_dir = os.path.join(args.save, "finetuned_patch", args.datetime)
        args.results_db = os.path.join(args.save_dir, args.results_db)

        os.makedirs(args.save_dir, exist_ok=True)

        # np.random.seed(args.seed)
        # print(f"=> Random Seed Set to {args.seed}")

        # ordering = list(np.random.permutation(len(args.eval_datasets) - 1))
        # args.eval_datasets = [args.eval_datasets[i] for i in ordering]
        
        # Order the checkpoints in the same order as the eval_datasets
        finetuned_checkpoints = [args.finetuned1_ckpt, args.finetuned2_ckpt]
        # dataset_to_ckpt = {}
        # for eval_dataset in args.eval_datasets[1:]:
        #     for ckpt in finetuned_checkpoints:
        #         if eval_dataset in ckpt:
        #             dataset_to_ckpt[eval_dataset] = ckpt
        #             break

        # finetuned_checkpoints = [dataset_to_ckpt[args.eval_datasets[i]] for i in range(len(finetuned_checkpoints))]

        finetuned_patch(args, finetuned_checkpoints)