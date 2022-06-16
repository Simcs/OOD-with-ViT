# -*- coding: utf-8 -*-
'''
Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
'''

from __future__ import print_function
from typing import Any, Dict, Tuple

from ml_collections.config_dict import ConfigDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import torchvision.transforms as transforms

from models import *
from models.vit import ViT

from datasets import OC_CIFAR10


class ViT_OC_CIFAR10_Trainer:
    
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.create_model()
        self.trainloader, self.testloader = self.create_dataloader()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=self.config.train.use_amp)
        self.criterion = nn.CrossEntropyLoss()
        
    def create_model(self) -> nn.Module:    
        model_name = self.config.model.name
        
        if model_name == 'ViT':
            # ViT for cifar10
            model = ViT(
                image_size = self.config.dataset.img_size,
                patch_size = self.config.model.patch_size,
                num_classes = len(self.config.dataset.in_distribution_class_indices),
                dim = self.config.model.dim_head,
                depth = self.config.model.depth,
                heads = self.config.model.n_heads,
                mlp_dim = self.config.model.dim_mlp,
                dropout = self.config.model.dropout,
                emb_dropout = self.config.model.emb_dropout,
            )
        else:
            raise NotImplementedError(f'Does not support model {model_name}')\
                
        if self.device == 'cuda':
            model = model.to(self.device)
            model = torch.nn.DataParallel(model) # make parallel
            cudnn.benchmark = True
            
        return model

    def create_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        dataset_root = self.config.dataset.root
        img_size = self.config.dataset.img_size
        in_distribution_class_indices = self.config.dataset.in_distribution_class_indices
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
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
        trainset = OC_CIFAR10(
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

        testset = OC_CIFAR10(
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
    
    def create_optimizer(self) -> Optimizer:
        optimizer_name = self.config.optimizer.name
        lr = self.config.optimizer.base_lr
        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        return optimizer
    
    def create_scheduler(self):
        # use cosine or reduce LR on Plateau scheduling
        scheduler_name = self.config.train.scheduler
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, 
                T_max=self.config.train.n_epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, 
                mode='min',
                factor=0.1,
                patience=3, 
                min_lr=1e-3*1e-5,
                verbose=True,
            )
        return scheduler
        
    def train(self):
        self.model.train()
        
        total_train_loss, n_correct, n_total = 0, 0, 0
        
        for batch_idx, (x, y) in enumerate(self.trainloader):
            x, y = x.to(self.device), y.to(self.device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=self.config.train.use_amp):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            n_total += y.size(0)
            n_correct += predicted.eq(y).sum().item()
            
        avg_train_loss = total_train_loss / (batch_idx + 1)
        train_accuracy = 100. * n_correct / n_total
        print(f'Train Loss: {avg_train_loss:.3f} | Train Acc: {train_accuracy:.3f}% ({n_correct}/{n_total})')

        return avg_train_loss
    
    def test(self):
        self.model.eval()
        
        total_test_loss, n_correct, n_total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.testloader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_test_loss += loss.item()
                _, predicted = outputs.max(1)
                n_total += y.size(0)
                n_correct += predicted.eq(y).sum().item()
        
            avg_test_loss = total_test_loss / (batch_idx + 1)
            test_accuracy = 100. * n_correct / n_total
            print(f'Test Loss: {avg_test_loss:.3f} | Test Acc: {test_accuracy:.3f}% ({n_correct}/{n_total})')
        
        return total_test_loss, test_accuracy
    
    def step_scheduler(self, validation_loss):
        if self.config.train.scheduler == 'cosine':
            self.scheduler.step()
        else:
            self.scheduler.step(validation_loss)

