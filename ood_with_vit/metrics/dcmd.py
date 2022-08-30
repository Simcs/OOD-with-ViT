from __future__ import annotations

from typing import List, Optional, Tuple
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance

import torch
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict

from ood_with_vit.utils import compute_logits, compute_penultimate_features
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor
from . import MaskMetric


class DCMD(MaskMetric):
    """
    Implementation of Difference of Classwise Mahalanobis Distance metric.
    """
    
    def __init__(
        self, 
        config: ConfigDict,
        model: torch.nn.Module,
        id_dataloader: DataLoader,
        feature_extractor: Optional[FeatureExtractor] = None,
        mask_method: str = 'top_ratio',
        mask_ratio: float = 0.3,
        mask_threshold: float = 0.9,
        head_fusion: str = 'max',
        discard_ratio: float = 0.9,
    ):
        super().__init__(
            config=config, 
            model=model, 
            mask_method=mask_method, 
            mask_ratio=mask_ratio, 
            mask_threshold=mask_threshold, 
            head_fusion=head_fusion, 
            discard_ratio=discard_ratio,
        )
        self.trainloader = id_dataloader
        self.feature_extractor = feature_extractor
        self.sample_means, self.precisions = self._compute_statistics()
        self.attention_masking.hook()

    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sample mean and classwise precisions (inverse of covariance)
        """
        self.model.eval()        
        with torch.no_grad():
            # compute penultimate features of each class
            class_to_features = [[] for _ in range(self.num_class)]
            for x, y in tqdm(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                penultimate_features = compute_penultimate_features(
                    config=self.config, 
                    model=self.model, 
                    imgs=x,
                    feature_extractor=self.feature_extractor,
                )
                for feature, label in zip(penultimate_features, y):
                    class_to_features[label.item()].append(feature.view(1, -1))
                
            for i in range(len(class_to_features)):
                class_to_features[i] = torch.cat(class_to_features[i], dim=0)
                
            # compute penultimate feature means of each class
            sample_means = [None] * self.num_class
            for i, feat in enumerate(class_to_features):
                sample_means[i] = torch.mean(feat, dim=0)

            # compute covariance matrix of each class
            precisions = []
            for list_feature, cls_mean in zip(class_to_features, sample_means):
                # group_lasso = EmpiricalCovariance(assume_centered=False)
                group_lasso = ShrunkCovariance(assume_centered=False)
                X = (list_feature - cls_mean).numpy()
                group_lasso.fit(X)
                precision = torch.from_numpy(group_lasso.precision_).float()
                precisions.append(precision)
            norms = [np.linalg.norm(precision) for precision in precisions]
            print('covairance norms:', norms)
        
        return sample_means, precisions

    def compute_img_ood_score(self, img: np.ndarray) -> float:
        """
        Compute DCMD based out-of-distrbution score given a test data.
        """
        self.model.eval()
        with torch.no_grad():
            self.attention_masking.disable_masking()
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            original_logit = compute_logits(self.config, self.model, img.unsqueeze(0))
            original_max_logit, original_pred = original_logit.max(dim=1)

            self.attention_masking.enable_masking()
            self.attention_masking.generate_mask(img)
            masked_logit = compute_logits(self.config, self.model, img.unsqueeze(0))
            masked_max_logit, masked_pred = masked_logit.max(dim=1)

        original_max_logit, original_pred = original_max_logit.item(), original_pred.item()
        masked_max_logit, masked_pred = masked_max_logit.item(), masked_pred.item()
        return original_max_logit - masked_max_logit
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            total_dcmd = []
            for x, y in tqdm(dataloader):
                original_gaussian_scores = []
                x, y = x.to(self.device), y.to(self.device)
                self.attention_masking.disable_masking()
                original_features = compute_penultimate_features(
                    config=self.config, 
                    model=self.model, 
                    imgs=x,
                    feature_extractor=self.feature_extractor
                )

                for sample_mean, prec in zip(self.sample_means, self.precisions):
                    zero_f = original_features - sample_mean
                    gau_term = torch.mm(torch.mm(zero_f, prec), zero_f.t()).diag()
                    original_gaussian_scores.append(gau_term.view(-1, 1))
                    
                original_gaussian_scores = torch.cat(original_gaussian_scores, dim=1)
                original_mahalanobis_distances, _ = original_gaussian_scores.min(dim=1)
                    
                self.attention_masking.generate_masks(x)

                self.attention_masking.enable_masking()
                masked_gaussian_scores = []
                masked_features = compute_penultimate_features(
                    config=self.config, 
                    model=self.model, 
                    imgs=x,
                    feature_extractor=self.feature_extractor
                )

                for sample_mean, prec in zip(self.sample_means, self.precisions):
                    zero_f = masked_features - sample_mean
                    gau_term = torch.mm(torch.mm(zero_f, prec), zero_f.t()).diag()
                    masked_gaussian_scores.append(gau_term.view(-1, 1))
                masked_gaussian_scores = torch.cat(masked_gaussian_scores, dim=1)
                masked_mahalanobis_distances, _ = masked_gaussian_scores.min(dim=1)

                for original_dist, masked_dist in zip(original_mahalanobis_distances, masked_mahalanobis_distances):
                    original_dist, masked_dist = original_dist.item(), masked_dist.item()
                    total_dcmd.append(original_dist - masked_dist)

        return total_dcmd