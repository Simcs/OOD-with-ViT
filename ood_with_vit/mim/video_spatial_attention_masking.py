from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ml_collections import ConfigDict
from ood_with_vit.utils.utils import compute_attention_maps

from ood_with_vit.visualizer import ViTAttentionRollout
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor
from ood_with_vit.mim.attention_masking import AttentionMaskingHooker


MASK_METHODS = [
    'top_ratio', # mask top K% attended patches
    'bottom_ratio', # mask botton K% attended patches
    'gt_threshold', # mask patches with attention score greather than threshold
    'lt_threshold', # mask patches with attention score lower than threshold
    'random', # randomly mask patches of the ratio
]

class VideoSpatialAttentionMaskingHooker(AttentionMaskingHooker):
    
    def mask_patch(self, module, input, output):
        if self.status == 'normal':
            return output

        assert self.masks is not None, 'mask should have been generated'
        w = torch.as_tensor(self.masks).to(self.device)
        w = rearrange(w, '(b t) h w -> b 1 t h w', b=output.size(0))
        w = w.bool()
        if self.mask_mode == 'zero':
            output = output.masked_fill(w, 0)
        elif self.mask_mode == 'minus_one':
            output = output.masked_fill(w, -1)
        elif self.mask_mode == 'average':
            masked_output = []
            for out, mask in zip(output, w):
                masked_output.append(out.masked_fill(mask, out.mean()))
            output = torch.stack(masked_output)
        # output = output * (1 - w)
        return output
    
    def _compute_rollout_attention_map(self, videos):
        self.attention_extractor.hook()
        attention_maps = compute_attention_maps(
            config=self.config,
            model=self.model,
            input=videos,
            feature_extractor=self.attention_extractor,
        )
        # TODO: '12' -> self.config.num_spatial_transformer_layers
        self.attention_maps = attention_maps[:12]
        self.attention_extractor.remove_hooks()
        
        rollout_attention_map = self.attention_rollout.spatial_rollout(self.attention_maps)
        return rollout_attention_map
    
    def generate_masks(self, videos, _precomputed_rollout_attention_maps=None):
        if _precomputed_rollout_attention_maps is None:
            rollout_attention_map = self._compute_rollout_attention_map(videos)
        else:
            rollout_attention_map = _precomputed_rollout_attention_maps
        masks = self.compute_attention_masks(rollout_attention_map)
        self.masks = masks
        return masks