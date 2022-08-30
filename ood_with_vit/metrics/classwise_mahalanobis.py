from typing import List, Optional, Tuple
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance

import torch
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict

from ood_with_vit.utils import compute_penultimate_features
from . import Metric

class ClasswiseMahalanobis(Metric):
    
    def __init__(
        self, 
        config: ConfigDict,
        model: torch.nn.Module,
        id_dataloader: DataLoader,
        feature_extractor: Optional[object] = None,
        _precomputed_statistics: Optional[object] = None,
    ):
        super().__init__(config, model)
        
        self.trainloader = id_dataloader
        self.feature_extractor = feature_extractor
        if _precomputed_statistics is None:
            self.statistics = self._compute_statistics()
        else:
            self.statistics = _precomputed_statistics
        self.sample_means, self.precisions = self.statistics
    
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
        Compute Mahalanobis distance based out-of-distrbution score given a test data.
        """
        self.model.eval()
        with torch.no_grad():
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            print(self.feature_extractor)
            feature = compute_penultimate_features(
                config=self.config, 
                model=self.model, 
                imgs=img.unsqueeze(0),
                feature_extractor=self.feature_extractor
            )
            
            gaussian_scores = []
            for sample_mean in self.sample_means:
                zero_f = feature - sample_mean
                gau_term = torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()
                gaussian_scores.append(gau_term.view(-1, 1))
            gaussian_scores = torch.cat(gaussian_scores, dim=1)
            mahalobis_distance, _ = gaussian_scores.min(dim=1)
            
        
        return mahalobis_distance.item()
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        closest_classes = []
        with torch.no_grad():
            total_mahalanobis_distances = []
            for x, y in tqdm(dataloader):
                gaussian_scores = []
                x, y = x.to(self.device), y.to(self.device)
                features = compute_penultimate_features(
                    config=self.config, 
                    model=self.model, 
                    imgs=x,
                    feature_extractor=self.feature_extractor
                )
                
                gaussian_scores = []
                for sample_mean, prec in zip(self.sample_means, self.precisions):
                    zero_f = features - sample_mean
                    gau_term = torch.mm(torch.mm(zero_f, prec), zero_f.t()).diag()
                    gaussian_scores.append(gau_term.view(-1, 1))
                gaussian_scores = torch.cat(gaussian_scores, dim=1)
                mahalanobis_distances, _ = gaussian_scores.min(dim=1)
                # [temp]: compute closest classes for debugging
                closest_classes.extend(_.numpy())
                total_mahalanobis_distances.extend(mahalanobis_distances.numpy())

        self.closest_classes = closest_classes
        self.total_mahalanobis_distances = total_mahalanobis_distances
        return total_mahalanobis_distances