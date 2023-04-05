import os
import time
import random
from copy import copy

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize, get_dataloaders
from src.datasets.registry import get_datasets
from eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head


import src.datasets as datasets

import wandb

def restrict_grad_dims(params, k=50):
    for param in params:
        if param.grad is not None:
            if len(param.shape) == 2:
                param.grad[:, k:] = 0.


def finetune(args):
    # Check if checkpoints already exist
    zs_path = os.path.join(args.save_dir, 'checkpoint_0.pt')  
    ft_path = os.path.join(args.save_dir, f'checkpoint_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert args.train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        print('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_heads = get_classification_head(args, args.train_dataset) # dict of dataset name to classification head

    model = ImageClassifier(image_encoder, classification_heads)

    model.freeze_head()

    if args.params_to_unfreeze is not None:
        model.freeze_all_except(args.params_to_unfreeze)

    preprocess_fn = model.train_preprocess
    print_every = 100

    datasets = get_datasets(
        args.train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = {name : len(dataset.train_loader) for name, dataset in datasets.items()}
    num_batches_total = sum(list(num_batches.values()))
    dataset_names = sorted(list(datasets.keys()))

    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    params = [p for p in model.parameters() if p.requires_grad]
    param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("="*80)
    print("Tuneable params:", param_names)
    print("="*80)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches_total)

    # Saving model
    if args.save is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        model_path = os.path.join(args.save_dir, f'checkpoint_0.pt')
        model.module.image_encoder.save(model_path)

    for epoch in range(args.epochs):
        model.train()
        model = model.cuda()

        data_loaders = get_dataloaders(
            datasets, is_train=True, args=args, image_encoder=None)

        num_batches_remaining = copy(num_batches)
        # for i, batch in enumerate(data_loader):
        for i in range(num_batches_total):
            # Sample a dataset proportional to the number of batches
            weights = [num_batches_remaining[d] for d in dataset_names]
            sampled_dataset = random.choices(dataset_names, weights=weights, k=1)[0]
            num_batches_remaining[sampled_dataset] -= 1

            # Get batch
            # batch = {name : next(data_loader) for name, data_loader in data_loaders.items()}
            batch = {sampled_dataset : next(data_loaders[sampled_dataset])}
            start_time = time.time()
            
            step = i + epoch * num_batches_total
            scheduler(step)
            optimizer.zero_grad()
            
            batch = {name : maybe_dictionarize(batch_) for name, batch_ in batch.items()}
            inputs = {name : batch_['images'].cuda() for name, batch_ in batch.items()}
            labels = {name: batch_['labels'].cuda() for name, batch_ in batch.items()}
            data_time = time.time() - start_time

            logits = model(inputs)

            losses = [loss_fn(logits[name], labels[name]) for name in logits]
            loss = sum(losses)
            # loss = loss_fn(logits, labels)

            if args.wandb:
                wandb.log({'train/loss_step':loss, 'step':step})

            loss.backward()

            if args.restrict_grad_dims:
                restrict_grad_dims(params, k=args.k)

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches_total
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches_total}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        image_encoder = model.module.image_encoder

        # Saving model
        if args.save is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, f'checkpoint_{epoch+1}.pt')
            image_encoder.save(model_path)
            optim_path = os.path.join(args.save_dir, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        args.current_epoch = epoch
        if args.eval_every_epoch:
            evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(args.save_dir, 'checkpoint_0.pt')  
        ft_path = os.path.join(args.save_dir, f'checkpoint_{args.epochs}.pt')    
        return zs_path, ft_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
