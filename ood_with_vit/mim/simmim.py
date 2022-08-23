import math
import random
import numpy as np

import torch
import torch.nn as nn

from ml_collections import ConfigDict


class MaskGenerator:
    def __init__(
        self, 
        input_size: int = 192, 
        mask_patch_size: int = 32, 
        model_patch_size: int = 4, 
        mask_ratio: float = 0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

class SimMIMHooker:
    def __init__(
        self,
        config: ConfigDict,
        model: nn.Module,
        patch_embedding_layer_name: str,
        mask_ratio: float = 0.6,
    ):
        self.model = model
        self.mask = None
        self.patch_embedding_layer_name = patch_embedding_layer_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: add mask_ratio to config
        self.mask_ratio = mask_ratio
        self.mask_generator = MaskGenerator(
            input_size=config.model.img_size,
            mask_patch_size=config.model.patch_size,
            model_patch_size=config.model.patch_size,
            mask_ratio=self.mask_ratio,
        )

    def hook(self):
        for name, module in self.model.named_modules():
            # for SimMIM based patch masking
            if self.patch_embedding_layer_name in name:
                module.register_forward_hook(self.mask_patch)
                print('name:', name)

    def mask_patch(self, module, input, output):
        # store generated mask
        self.mask = self.mask_generator()
        mask = torch.as_tensor(self.mask).to(self.device)
        w = mask.unsqueeze(0).flatten(1).unsqueeze(-1)
        output = output * (1 - w)
        return output

    def get_mask(self):
        return self.mask
        
    def __call__(self, input):
        with torch.no_grad():
            output = self.model(input)
        return output
