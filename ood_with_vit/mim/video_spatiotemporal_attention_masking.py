from typing import TYPE_CHECKING, Optional

import numpy as np

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

class VideoSpatiotemporalAttentionMaskingHooker(AttentionMaskingHooker):
    
    def __init__(
        self,
        config: ConfigDict,
        model: nn.Module,
        attention_extractor: FeatureExtractor,
        patch_embedding_layer_name: str,
        mask_mode: str,
        spatial_mask_method: str,
        temporal_mask_method: str,
        spatial_mask_ratio: Optional[float] = None,
        spatial_mask_threshold: Optional[float] = None,
        temporal_mask_ratio: Optional[float] = None,
        temporal_mask_threshold: Optional[float] = None,
        head_fusion: str = 'max',
        discard_ratio: float = 0.9,
    ):
        self.config = config
        self.model = model
        self.attention_extractor = attention_extractor
        self.patch_embedding_layer_name = patch_embedding_layer_name
        self.mask_mode = mask_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status = 'normal'

        self.spatial_attention_maps = None
        self.temporal_attention_maps = None
        self.masks = None
        
        self.attention_rollout = ViTAttentionRollout(
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )
        self.spatial_mask_method = spatial_mask_method
        self.spatial_mask_ratio = spatial_mask_ratio
        self.spatial_mask_threshold = spatial_mask_threshold
        self.temporal_mask_method = temporal_mask_method
        self.temporal_mask_ratio = temporal_mask_ratio
        self.temporal_mask_threshold = temporal_mask_threshold

        self.handles = []
    
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
        self.spatial_attention_maps = attention_maps[:12]
        self.temporal_attention_maps = attention_maps[12:]
        self.attention_extractor.remove_hooks()
        
        rollout_spatial_attention_map = self.attention_rollout.spatial_rollout(self.spatial_attention_maps)
        rollout_temporal_attention_map = self.attention_rollout.temporal_rollout(self.temporal_attention_maps)
        return rollout_spatial_attention_map, rollout_temporal_attention_map

    def compute_attention_masks(self, spatial_attention_maps, temporal_attention_maps):
        # generate masks from attention_map computed by AttentionRollout
        temporal_attention_maps = torch.from_numpy(temporal_attention_maps)
        spatial_attention_maps = torch.from_numpy(spatial_attention_maps)
        temporal_flat = temporal_attention_maps.view(temporal_attention_maps.size(0), -1)
        spatial_flat = spatial_attention_maps.view(spatial_attention_maps.size(0), -1)

        def _masks_from_mask_indices(attn_maps, mask_indices):
            masks = torch.zeros(attn_maps.shape)
            flat_mask = masks.view(masks.size(0), -1)
            flat_mask.scatter_(1, mask_indices, 1.)
            return masks

        # 1. compute temporal masks
        if self.temporal_mask_method == 'top_ratio':
            mask_count = int(temporal_flat.size(-1) * self.temporal_mask_ratio)
            _, mask_idx = temporal_flat.topk(mask_count, dim=-1, largest=True)
            masks = _masks_from_mask_indices(temporal_attention_maps, mask_idx)
        elif self.temporal_mask_method == 'bottom_ratio':
            mask_count = int(temporal_flat.size(-1) * self.temporal_mask_ratio)
            _, mask_idx = temporal_flat.topk(mask_count, dim=-1, largest=False)
            masks = _masks_from_mask_indices(temporal_attention_maps, mask_idx)
        elif self.temporal_mask_method == 'gt_threshold':
            masks = torch.where(temporal_attention_maps > self.temporal_mask_threshold, 1., 0.)
        elif self.temporal_mask_method == 'lt_threshold':
            masks = torch.where(temporal_attention_maps < self.temporal_mask_threshold, 1., 0.)
        elif self.temporal_mask_method == 'random':
            mask_count = int(temporal_flat.size(-1) * self.temporal_mask_ratio)
            mask_idx = []
            for _ in range(temporal_flat.size(0)):
                mask_idx.append(torch.randperm(temporal_flat.size(-1))[:mask_count])
            mask_idx = torch.stack(mask_idx)
            masks = _masks_from_mask_indices(temporal_attention_maps, mask_idx)
        temporal_masks = masks
        
        # 2. compute spatial masks
        if self.spatial_mask_method == 'top_ratio':
            mask_count = int(spatial_flat.size(-1) * self.spatial_mask_ratio)
            _, mask_idx = spatial_flat.topk(mask_count, dim=-1, largest=True)
            masks = _masks_from_mask_indices(spatial_attention_maps, mask_idx)
        elif self.spatial_mask_method == 'bottom_ratio':
            mask_count = int(spatial_flat.size(-1) * self.spatial_mask_ratio)
            _, mask_idx = spatial_flat.topk(mask_count, dim=-1, largest=False)
            masks = _masks_from_mask_indices(spatial_attention_maps, mask_idx)
        elif self.spatial_mask_method == 'gt_threshold':
            masks = torch.where(spatial_attention_maps > self.spatial_mask_threshold, 1., 0.)
        elif self.spatial_mask_method == 'lt_threshold':
            masks = torch.where(spatial_attention_maps < self.spatial_mask_threshold, 1., 0.)
        elif self.spatial_mask_method == 'random':
            mask_count = int(spatial_flat.size(-1) * self.spatial_mask_ratio)
            mask_idx = []
            for _ in range(spatial_flat.size(0)):
                mask_idx.append(torch.randperm(spatial_flat.size(-1))[:mask_count])
            mask_idx = torch.stack(mask_idx)
            masks = _masks_from_mask_indices(spatial_attention_maps, mask_idx)
        spatial_masks = masks       
        
        # 3. compute spatiotemporal masks
        spatiotemporal_masks = spatial_masks * temporal_masks.view(spatial_masks.size(0), 1, 1)        
        # print('temporal mask[0]:', temporal_masks[0])
        # print('spatial maks[0]:', spatial_masks[:8], spatial_masks[:8].shape)
        # print('spatiotemporal masks[0]:', spatiotemporal_masks[:8], spatiotemporal_masks[:8].shape)
        
        # print('n spatial:', np.where(spatial_masks == 1)[0].shape[0])
        # print('n temporal:', np.where(temporal_masks == 1)[0].shape[0])
        # print('n spatiotemporal:', np.where(spatiotemporal_masks == 1)[0].shape[0])
        return spatiotemporal_masks.numpy()
     
    def generate_masks(
        self,
        videos,
        _precomputed_rollout_spatial_attention_maps=None,
        _precomputed_rollout_temporal_attention_maps=None,
    ):
        if _precomputed_rollout_spatial_attention_maps is None or \
            _precomputed_rollout_temporal_attention_maps is None:
            rollout_spatial_attention_map, rollout_temporal_attention_map \
                = self._compute_rollout_attention_map(videos)
        else:
            rollout_spatial_attention_map = _precomputed_rollout_spatial_attention_maps
            rollout_temporal_attention_map = _precomputed_rollout_temporal_attention_maps
        masks = self.compute_attention_masks(
            rollout_spatial_attention_map,
            rollout_temporal_attention_map,
        )
        self.masks = masks
        return masks