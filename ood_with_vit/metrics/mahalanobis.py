
from typing import List, Tuple
from PIL import Image

import numpy as np
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ml_collections.config_dict import ConfigDict

from ood_with_vit.datasets import OOD_CIFAR10
from ood_with_vit.utils import compute_penultimate_features, compute_logits


class Mahalanobis:
    
    def __init__(self, 
                 config: ConfigDict,
                 model: torch.nn.Module,
                 id_dataloader: DataLoader):
        self.config = config
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_class = config.dataset.n_class
        
        self.trainloader = id_dataloader
        self.sample_means, self.precision = self._compute_statistics()
    
    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sample mean and precision (inverse of covariance)
        return: 
            sample_class_man:
            precision
        """
        self.model.eval()
        
        # group_lasso = EmpiricalCovariance(assume_centered=False)
        group_lasso = ShrunkCovariance(assume_centered=False)
        
        with torch.no_grad():
            # compute penultimate features of each class
            class_to_features = [[] for _ in range(self.num_class)]
            for x, y in tqdm(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                penultimate_features = compute_penultimate_features(self.config, self.model, x)
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
        Compute Mahalanobis distance based out-of-distrbution score given a test data.
        """
        self.model.eval()
        with torch.no_grad():
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            feature = compute_penultimate_features(self.config, self.model, img.unsqueeze(0))
            
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
        with torch.no_grad():
            total_mahalanobis_distances = []
            for x, y in tqdm(dataloader):
                gaussian_scores = []
                x, y = x.to(self.device), y.to(self.device)
                features = compute_penultimate_features(self.config, self.model, x)
                
                for sample_mean in self.sample_means:
                    zero_f = features - sample_mean
                    gau_term = torch.mm(torch.mm(zero_f, self.precision), zero_f.t()).diag()
                    gaussian_scores.append(gau_term.view(-1, 1))
            
                gaussian_scores = torch.cat(gaussian_scores, dim=1)
                mahalanobis_distances, _ = gaussian_scores.min(dim=1)
                total_mahalanobis_distances.extend(mahalanobis_distances.numpy())
            
        return total_mahalanobis_distances