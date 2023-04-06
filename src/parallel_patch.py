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

    patching_ckpts = [args.mnist_ckpt, args.cars_ckpt, args.kitti_ckpt, args.svhn_ckpt, args.gtsrb_ckpt]
    parallel_patch(args, args.zeroshot_ckpt, patching_ckpts)