
from typing import List, Tuple
from PIL import Image

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict

from ood_with_vit.utils import compute_logits
from . import Metric


class SML(Metric):
    
    def __init__(
        self, 
        config: ConfigDict,
        model: torch.nn.Module,
        id_dataloader: DataLoader):
        super().__init__(config, model)
        
        self.trainloader = id_dataloader
        self.max_logit_means, self.max_logit_stds = self._compute_statistics()
    
    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sample mean and precision (inverse of covariance)
        return: 
            sample_class_man:
            precision
        """
        self.model.eval()        
        with torch.no_grad():
            # compute penultimate features of each class
            class_to_max_logits = [[] for _ in range(self.num_class)]
            for x, y in tqdm(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                logits = compute_logits(self.config, self.model, x)
                max_logits, preds = logits.max(dim=1)
                
                for max_logit, pred, label in zip(max_logits, preds, y):
                    # ignore misclassified examples
                    if pred.item() != label.item():
                        continue
                    class_to_max_logits[label].append(max_logit.item())
                    
            max_logit_means, max_logit_stds = [], []
            for max_logits in class_to_max_logits:
                mean, std = np.mean(max_logits), np.std(max_logits)
                max_logit_means.append(mean)
                max_logit_stds.append(std)
        
        return max_logit_means, max_logit_stds
        
    def compute_img_ood_score(self, img: np.ndarray) -> float:
        """
        Compute SML based out-of-distrbution score given a test data.
        """
        self.model.eval()
        with torch.no_grad():
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            logit = compute_logits(self.config, self.model, img.unsqueeze(0))
            max_logit, pred = logit.max(dim=1)
        max_logit, pred = max_logit.item(), pred.item()
        mean, std = self.max_logit_means[pred], self.max_logit_stds[pred]
        return -(max_logit - mean) / std
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            total_sml = []
            for x, y in tqdm(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                logits = compute_logits(self.config, self.model, x)
                max_logits, preds = logits.max(dim=1)
        
                for max_logit, pred in zip(max_logits, preds):
                    max_logit, pred = max_logit.item(), pred.item()
                    mean, std = self.max_logit_means[pred], self.max_logit_stds[pred]
                    sml = -(max_logit - mean) / std
                    total_sml.append(sml)
        return total_sml