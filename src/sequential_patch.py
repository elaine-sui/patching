import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from src.eval import evaluate
from src.modeling import ImageEncoder
from src.args import parse_arguments

def sequential_patch(args, zeroshot_checkpoint, finetuned_checkpoints):
    # Load models
    zeroshot = ImageEncoder.load(zeroshot_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    del zeroshot

    # Eval datasets
    all_eval_datasets = args.eval_datasets

    alphas = args.alpha
    for i, ckpt in enumerate(finetuned_checkpoints):
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
            finetuned.save(os.path.join(args.save_dir, f'sequential_patch_it_{i}_alpha={alpha:.3f}.pt'))

            # evaluate
            _, avg_acc = evaluate(finetuned, args)
            
            avg_accs.append(avg_acc)
        
        # Best interpolation is the new zero-shot
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
    args = parse_arguments()
    args.save_dir = os.path.join(args.save, "sequential_patch")
    args.results_db = os.path.join(args.save_dir, args.results_db)

    os.makedirs(args.save_dir, exist_ok=True)
    zeroshot_checkpoint = "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_04-17_52_45/checkpoint_0.pt"
    finetuned_checkpoints = [
        "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_04-17_52_45/checkpoint_5.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/CarsVal/2023_04_04-17_52_40/checkpoint_35.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/KITTIVal/2023_04_04-17_52_40/checkpoint_40.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/SVHNVal/2023_04_04-17_52_45/checkpoint_4.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/GTSRBVal/2023_04_04-17_52_45/checkpoint_9.pt"
    ]

    sequential_patch(args, zeroshot_checkpoint, finetuned_checkpoints)