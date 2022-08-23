from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import ConfigDict
from ood_with_vit.utils.utils import compute_attention_maps

from ood_with_vit.visualizer import ViTAttentionRollout
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor


MASK_METHODS = [
    'top_ratio', # mask top K% attended patches
    'bottom_ratio', # mask botton K% attended patches
    'gt_threshold', # mask patches with attention score greather than threshold
    'lt_threshold', # mask patches with attention score lower than threshold
    'random', # randomly mask patches of the ratio
]

class AttentionMaskingHooker:

    def __init__(
        self,
        config: ConfigDict,
        model: nn.Module,
        attention_extractor: FeatureExtractor,
        patch_embedding_layer_name: str,
        mask_method: str,
        mask_ratio: Optional[float] = None,
        mask_threshold: Optional[float] = None,
        head_fusion: str = 'max',
        discard_ratio: float = 0.9,
    ):
        assert mask_method in MASK_METHODS, 'invalid mask methods'
        assert not(mask_ratio is None and mask_threshold is None), \
            'neither mask_ratio nor mask_threshold are None'

        self.config = config
        self.model = model
        self.attention_extractor = attention_extractor
        self.patch_embedding_layer_name = patch_embedding_layer_name
        self.mask_method = mask_method
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status = 'normal'

        self.attention_maps = None
        self.masks = None
        
        self.attention_rollout = ViTAttentionRollout(
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )
        self.mask_ratio = mask_ratio
        self.mask_threshold = mask_threshold

        self.handles = []

    def enable_masking(self):
        self.status = 'masking'
    
    def disable_masking(self):
        self.status = 'normal'

    def switch_status(self, status):
        assert status in ['normal', 'masking']
        self.status = status
        
    def hook(self):
        for name, module in self.model.named_modules():
            if self.patch_embedding_layer_name in name:
                handle = module.register_forward_hook(self.mask_patch)
                self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def mask_patch(self, module, input, output):
        if self.status == 'normal':
            return output

        assert self.masks is not None, 'mask should have been generated'
        masks = torch.as_tensor(self.masks).to(self.device)
        w = masks.view(masks.size(0), -1).unsqueeze(-1)
        # w = masks.flatten(1).unsqueeze(-1)
        # print('w:', w.shape)
        # mask_idx = torch.nonzero(w[0, :, 0] == 1).squeeze()
        # print(mask_idx, mask_idx.shape)
        # print('before mask:', output.shape, w.shape)
        # print(output[0, mask_idx[0], :])
        output = output * (1 - w)
        # print(output.shape, w.shape)
        # print('after mask:', output.shape)
        # print(output[0, mask_idx[0], :])
        return output

    def mask_image(self, imgs):
        b, c, h, w = imgs.size()
        masks = torch.as_tensor(self.masks).to(self.device)
        masks = masks.unsqueeze(1) # add channel axis
        masks = F.interpolate(masks, size=(h, w))
        # masks = masks.expand(b, c, h, w)
        return imgs * (1 - masks)
    
    def generate_masks(self, imgs):
        self.attention_extractor.hook()
        attention_maps = compute_attention_maps(
            config=self.config,
            model=self.model,
            imgs=imgs,
            feature_extractor=self.attention_extractor,
        )
        self.attention_maps = attention_maps
        self.attention_extractor.remove_hooks()

        rollout_attention_map = self.attention_rollout.rollout(attention_maps)
        masks = self.compute_attention_masks(rollout_attention_map)

        self.masks = masks
        return masks
    
    def compute_attention_masks(self, attention_maps):
        # generate masks from attention_map computed by AttentionRollout
        attention_maps = torch.from_numpy(attention_maps)
        flat = attention_maps.view(attention_maps.size(0), -1)

        def _masks_from_mask_indices(mask_indices):
            masks = torch.zeros(attention_maps.shape)
            flat_mask = masks.view(masks.size(0), -1)
            flat_mask.scatter_(1, mask_indices, 1.)
            return masks

        if self.mask_method == 'top_ratio':
            mask_count = int(flat.size(-1) * self.mask_ratio)
            _, mask_idx = flat.topk(mask_count, dim=-1, largest=True)
            masks = _masks_from_mask_indices(mask_idx)
        elif self.mask_method == 'bottom_ratio':
            mask_count = int(flat.size(-1) * self.mask_ratio)
            _, mask_idx = flat.topk(mask_count, dim=-1, largest=False)
            masks = _masks_from_mask_indices(mask_idx)
        elif self.mask_method == 'gt_threshold':
            masks = torch.where(attention_maps > self.mask_threshold, 1., 0.)
        elif self.mask_method == 'lt_threshold':
            masks = torch.where(attention_maps < self.mask_threshold, 1., 0.)
        elif self.mask_method == 'random':
            mask_count = int(flat.size(-1) * self.mask_ratio)
            mask_idx = []
            for _ in range(flat.size(0)):
                mask_idx.append(torch.randperm(flat.size(-1))[:mask_count])
            mask_idx = torch.stack(mask_idx)
            masks = _masks_from_mask_indices(mask_idx)
        
        # if self.mask_metric == 'ratio':
        #     if self.mask_method == 'max':
        #         _, mask_idx = flat.topk(mask_count, dim=-1, largest=True)
        #     elif self.mask_method == 'min':
        #         _, mask_idx = flat.topk(mask_count, dim=-1, largest=False)
        #     elif self.mask_method == 'random':
        #         mask_idx = []
        #         for _ in range(flat.size(0)):
        #             mask_idx.append(torch.randperm(flat.size(-1))[:mask_count])
        #         mask_idx = torch.stack(mask_idx)   
        #     else:
        #         raise ValueError('Invalid attention masking type.')

        #     masks = torch.zeros(attention_maps.shape)
        #     flat_mask = masks.view(masks.size(0), -1)
        #     flat_mask.scatter_(1, mask_idx, 1.)

        # elif self.mask_metric == 'threshold':
        #     if self.mask_method == 'max':
        #         masks = torch.where(attention_maps > self.mask_threshold, 1., 0.)
        #     elif self.mask_method == 'min':
        #         masks = torch.where(attention_maps < self.mask_threshold, 1., 0.)
        #     else:
        #         raise ValueError('Invalid attention masking type.')
        # else:
        #     raise ValueError('Invalid attention masking type.')
        
        # flat_mask = masks.view(masks.size(0), -1)
        # for i in range(flat_mask.size(0)):
        #     flat_mask[i, mask_idx[i]] = 1
        return masks.numpy()