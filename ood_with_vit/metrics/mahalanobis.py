
from PIL import Image

import numpy as np
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
                 model: torch.nn.Module):
        self.config = config
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_class = len(config.dataset.in_distribution_class_indices)
        
        self.trainloader = self._create_dataloader()
        
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        img_size = self.config.model.img_size
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        
        self.sample_means, self.precision = self._compute_statistics()
        
    def _create_dataloader(self) -> DataLoader:
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        dataset_root = self.config.dataset.root
        img_size = self.config.model.img_size
        in_distribution_class_indices = self.config.dataset.in_distribution_class_indices
        
        transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        
        trainset = OOD_CIFAR10(
            root=dataset_root,
            in_distribution_class_indices=in_distribution_class_indices, 
            train=True, 
            download=True, 
            transform=transform_train
        )
        trainloader = DataLoader(
            dataset=trainset, 
            batch_size=self.config.train.batch_size, 
            shuffle=True, 
            num_workers=8
        )
        
        return trainloader
    
    def _compute_statistics(self):
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
            for x, y in self.trainloader:
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
        
    def compute_ood_score(self, img):
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