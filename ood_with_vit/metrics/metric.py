from typing import List, Optional

import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict

from ood_with_vit.visualizer.feature_extractor import FeatureExtractor
from ood_with_vit.mim.attention_masking import AttentionMaskingHooker


class Metric:
    
    def __init__(
        self, 
        config: ConfigDict,
        model: torch.nn.Module,
    ):
        self.config = config
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_class = config.dataset.n_class

        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        img_size = self.config.model.img_size
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        
    def compute_img_ood_score(self, img: np.ndarray) -> float:
        raise NotImplementedError()
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        raise NotImplementedError()


class MaskMetric(Metric):
    
    def __init__(
        self, 
        config: ConfigDict,
        model: torch.nn.Module,
        mask_method: str = 'top_ratio',
        mask_ratio: float = 0.3,
        mask_threshold: float = 0.9,
        head_fusion: str = 'max',
        discard_ratio: float = 0.9,
    ):
        super().__init__(config, model)

        attention_extractor = FeatureExtractor(
            model=model,
            layer_name=config.model.layer_name.attention,
        )
        self.attention_masking = AttentionMaskingHooker(
            config=config,
            model=model,
            attention_extractor=attention_extractor,
            patch_embedding_layer_name='patch_embed.norm',
            mask_method=mask_method,
            mask_ratio=mask_ratio,
            mask_threshold=mask_threshold,
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )
        
    def compute_img_ood_score(self, img: np.ndarray) -> float:
        raise NotImplementedError()
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        raise NotImplementedError()