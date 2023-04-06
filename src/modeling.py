import torch

import clip.clip as clip

from src import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = classification_heads # dictionary of dataset to classification head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for dataset_name, classification_head in self.classification_heads.items():
            classification_head.weight.requires_grad_(False)
            classification_head.bias.requires_grad_(False)

    def freeze_all_except(self, params_to_unfreeze=['last']):

        def has_name_starts_with_param(name, params_lst):
            for p in params_lst:
                if name.startswith(p):
                    return True
            
            return False

        print([name for name, _ in self.image_encoder.named_parameters()])
        
        if 'last' in params_to_unfreeze:
            params_to_unfreeze.remove('last')
            params_to_unfreeze.append('model.token_embedding.weight')
            params_to_unfreeze.extend(['model.visual.ln_post.weight', 'model.visual.ln_post.bias', 'model.ln_final.weight', 'model.ln_final.bias'])
            # params_to_unfreeze.extend(['model.visual.transformer.resblocks.11.mlp.c_proj.weight', 'model.visual.transformer.resblocks.11.mlp.c_proj.bias', 'model.visual.transformer.resblocks.11.ln_2.weight'])
        elif 'low' in params_to_unfreeze:
            params_to_unfreeze.remove('low')
            params_to_unfreeze.append('model.visual.transformer.resblocks.0')
        elif 'middle' in params_to_unfreeze:
            params_to_unfreeze.remove('middle')
            params_to_unfreeze.append('model.visual.transformer.resblocks.5')

        for name, param in self.image_encoder.named_parameters():
            to_unfreeze = has_name_starts_with_param(name, params_to_unfreeze)
            param.requires_grad_(to_unfreeze)
            if to_unfreeze:
                param.retain_grad()

    def forward(self, inputs):
        features = {dataset_name : self.image_encoder(inputs_) for dataset_name, inputs_ in inputs.items()}
        outputs = {}
        for dataset_name, features_ in features.items():
            outputs[dataset_name] = self.classification_heads[dataset_name](features_)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
