from typing import Tuple

from ml_collections.config_dict import ConfigDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from torchvision.datasets import CIFAR100
from . import BaseTrainer

class ViT_Finetune_CIFAR100_Trainer(BaseTrainer):
    
    def __init__(self, config: ConfigDict):
        super().__init__(config)
        
    def _create_model(self) -> nn.Module:    
        repo = self.config.model.repo
        model_name = self.config.model.pretrained_model
        model = torch.hub.load(
            repo_or_dir=repo,
            model=model_name,
            pretrained=True,
        )
        n_class = self.config.dataset.n_class
        model.head = nn.Linear(model.head.in_features, n_class)
        if self.device == 'cuda':
            model = model.to(self.device)
            model = torch.nn.DataParallel(model) # make parallel
            cudnn.benchmark = True
            
        return model
    
    def _create_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        dataset_root = self.config.dataset.root
        img_size = self.config.model.img_size
        
        print('==> Preparing data..')        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        trainset = CIFAR100(
            root=dataset_root,
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
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        testset = CIFAR100(
            root=dataset_root, 
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