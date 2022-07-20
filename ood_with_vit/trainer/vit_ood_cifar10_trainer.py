from typing import Tuple

from ml_collections.config_dict import ConfigDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import torchvision.transforms as transforms

from ood_with_vit.models.vit import ViT
from ood_with_vit.datasets import OOD_CIFAR10
from . import OOD_CIFAR10_Trainer


class ViT_OOD_CIFAR10_Trainer(OOD_CIFAR10_Trainer):
    
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        
    def _create_model(self) -> nn.Module:    
        model_name = self.config.model.name
        
        if model_name == 'custom-vit':
            # ViT for cifar10
            model = ViT(
                image_size=self.config.model.img_size,
                patch_size=self.config.model.patch_size,
                num_classes=len(self.config.dataset.in_distribution_class_indices),
                dim=self.config.model.dim_head,
                depth=self.config.model.depth,
                heads=self.config.model.n_heads,
                mlp_dim=self.config.model.dim_mlp,
                dropout=self.config.model.dropout,
                emb_dropout=self.config.model.emb_dropout,
                visualize=False,
            )
        else:
            raise NotImplementedError(f'Does not support model {model_name}')\
                
        if self.device == 'cuda':
            model = model.to(self.device)
            model = torch.nn.DataParallel(model) # make parallel
            cudnn.benchmark = True
            
        return model

    def _create_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        dataset_root = self.config.dataset.root
        img_size = self.config.model.img_size
        in_distribution_class_indices = self.config.dataset.in_distribution_class_indices
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        
        print('==> Preparing data..')
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

        testset = OOD_CIFAR10(
            root=dataset_root, 
            in_distribution_class_indices=in_distribution_class_indices, 
            train=False, 
            download=True, 
            transform=transform_test
        )
        testloader = DataLoader(
            dataset=testset, 
            batch_size=self.config.eval.batch_size, 
            shuffle=False, 
            num_workers=8
        )
        return trainloader, testloader