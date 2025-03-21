import os
import json

import torch

from src import utils
from src.utils import LabelSmoothing
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier

from src.datasets.registry import get_dataset

import wandb


def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, [dataset_name])
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)
    batched_data = enumerate(dataloader)
    device = args.device

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        total_loss = 0.
        for i, data in batched_data:
            data = maybe_dictionarize(data)
            x = {dataset_name : data['images'].to(device)}
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model, device=device)

            losses = [loss_fn(logits[name], y) for name in logits]
            total_loss += sum(losses)

            pred = logits[dataset_name].argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}

    if args.wandb:
        wandb.log({f'val/{dataset_name}_top1_acc': top1, f'val/{dataset_name}_loss': total_loss})
    
    return metrics

def evaluate(image_encoder, args):

    avg_acc = 0.
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            avg_acc += results['top1']
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info, avg_acc / len(args.eval_datasets)