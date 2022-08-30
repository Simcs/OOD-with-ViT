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


class MMD(MaskMetric):
    """
    Implementation of Masked Mahalanobis Distance
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
        _precomputed_statistics: Optional[object] = None,
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
        if _precomputed_statistics is None:
            self.statistics = self._compute_statistics()
        else:
            self.statistics = _precomputed_statistics
        self.sample_means, self.precision = self.statistics
        self.attention_masking.hook()

    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sample mean and precision (inverse of covariance)
        return: 
            sample_class_man:
            precision
        """
        self.model.eval()     
        
        group_lasso = EmpiricalCovariance(assume_centered=False)
        # group_lasso = ShrunkCovariance(assume_centered=False)   
        
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

            # compute covariance matrix of penultimate features
            X = []
            for list_feature, cls_mean in zip(class_to_features, sample_means):
                X.append(list_feature - cls_mean)
            X = torch.cat(X, dim=0).numpy()
            group_lasso.fit(X)
            precision = torch.from_numpy(group_lasso.precision_).float()
            print('covariance norm:', np.linalg.norm(group_lasso.precision_))
        
        return sample_means, precision

    def compute_img_ood_score(self, img: np.ndarray) -> float:
        """
        Compute MMD based out-of-distrbution score given a test data.
        """
        self.model.eval()
        with torch.no_grad():
            self.attention_masking.disable_masking()
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            
            self.attention_masking.disable_masking()
            self.attention_masking.generate_masks(img.unsqueeze(0))
            
            self.attention_masking.enable_masking()
            masked_gaussian_scores = []
            masked_features = compute_penultimate_features(
                config=self.config, 
                model=self.model, 
                imgs=img.unsqueeze(0),
                feature_extractor=self.feature_extractor
            )

            for sample_mean in self.sample_means:
                zero_f = masked_features - sample_mean
                gau_term = torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()
                masked_gaussian_scores.append(gau_term.view(-1, 1))
            masked_gaussian_scores = torch.cat(masked_gaussian_scores, dim=1)
            masked_mahalanobis_distance, _ = masked_gaussian_scores.min(dim=1)

        return masked_mahalanobis_distance.item()
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            total_mmd = []
            for x, y in tqdm(dataloader):
                x, y = x.to(self.device), y.to(self.device)

                self.attention_masking.disable_masking()
                self.attention_masking.generate_masks(x)

                self.attention_masking.enable_masking()
                masked_gaussian_scores = []
                masked_features = compute_penultimate_features(
                    config=self.config, 
                    model=self.model, 
                    imgs=x,
                    feature_extractor=self.feature_extractor
                )

                for sample_mean in self.sample_means:
                    zero_f = masked_features - sample_mean
                    gau_term = torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()
                    masked_gaussian_scores.append(gau_term.view(-1, 1))
                masked_gaussian_scores = torch.cat(masked_gaussian_scores, dim=1)
                masked_mahalanobis_distances, _ = masked_gaussian_scores.min(dim=1)

                total_mmd.extend(masked_mahalanobis_distances.numpy())
            
        return total_mmd