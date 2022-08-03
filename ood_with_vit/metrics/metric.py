from typing import List

import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict


class Metric:
    
    def __init__(self, 
                 config: ConfigDict,
                 model: torch.nn.Module):
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