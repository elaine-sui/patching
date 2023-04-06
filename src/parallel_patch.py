import os
import sys
from datetime import datetime
import wandb

sys.path.append(os.getcwd())

from src.eval import evaluate
from src.modeling import ImageEncoder
from src.args import parse_arguments

def wandb_init(args):
    model_name_safe = args.model.replace('/', '-')

    if len(args.eval_datasets) == 1:
        eval_dataset_str = args.eval_datasets[0]
    else:
        eval_dataset_str = "parallel_" + "_".join(args.eval_datasets)

    args.exp_name = model_name_safe
    if args.params_to_unfreeze is not None:
        unfreeze_str = '_'.join(args.params_to_unfreeze)
        args.exp_name += f"_unfreeze_{unfreeze_str}"
    
    if args.restrict_grad_dims:
        args.exp_name += f"_restrict_k_{args.k}_dims"

    args.exp_name += f"_{eval_dataset_str}/{args.datetime}"

    wandb.init(
        name=args.exp_name,
        project="patching",
        config=args
    )

def parallel_patch(args, zeroshot_checkpoint, finetuned_checkpoints):
    # Wandb init
    if args.wandb:
        wandb_init(args)
    
    args.wandb = False # Don't log the actual metrics (just log console)

    # Load models
    zeroshot = ImageEncoder.load(zeroshot_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    del zeroshot

    theta_1 = {}
    for ckpt in finetuned_checkpoints:
        finetuned = ImageEncoder.load(ckpt)

        for k, v in finetuned.state_dict().items():
            if k not in theta_1:
                theta_1[k] = v.clone()
            else:
                theta_1[k] += v.clone()
    
    # Average the weights
    theta_1 = {k:v / len(finetuned_checkpoints) for k, v in theta_1.items()}

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
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
        finetuned.save(os.path.join(args.save_dir, f'alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)

if __name__ == '__main__':
    args = parse_arguments()
    args.datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    args.save_dir = os.path.join(args.save, "parallel_patch", args.datetime)
    args.results_db = os.path.join(args.save_dir, args.results_db)

    patching_ckpts = [args.mnist_ckpt, args.cars_ckpt, args.kitti_ckpt, args.svhn_ckpt, args.gtsrb_ckpt]
    parallel_patch(args, args.zeroshot_ckpt, patching_ckpts)