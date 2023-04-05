import os
import sys

sys.path.append(os.getcwd())

from src.eval import evaluate
from src.modeling import ImageEncoder
from src.args import parse_arguments

def parallel_patch(args, zeroshot_checkpoint, finetuned_checkpoints):
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
        finetuned.save(os.path.join(args.save, "parallel_patched", f'parallel_patched_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)

if __name__ == '__main__':
    args = parse_arguments()
    args.results_db = os.path.join(args.save, "parallel_patched", args.results_db)
    zeroshot_checkpoint = "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_04-17_52_45/checkpoint_0.pt"
    finetuned_checkpoints = [
        "/pasteur/u/esui/patching/models/patch/ViTB32/MNISTVal/2023_04_04-17_52_45/checkpoint_5.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/CarsVal/2023_04_04-17_52_40/checkpoint_35.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/KITTIVal/2023_04_04-17_52_40/checkpoint_40.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/SVHNVal/2023_04_04-17_52_45/checkpoint_4.pt",
        "/pasteur/u/esui/patching/models/patch/ViTB32/GTSRBVal/2023_04_04-17_52_45/checkpoint_9.pt"
    ]

    parallel_patch(args, zeroshot_checkpoint, finetuned_checkpoints)