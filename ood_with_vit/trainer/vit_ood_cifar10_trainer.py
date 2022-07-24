from typing import Tuple

from ml_collections.config_dict import ConfigDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from ood_with_vit.models.vit import ViT
from ood_with_vit.datasets import OOD_CIFAR10
from . import BaseTrainer


class ViT_OOD_CIFAR10_Trainer(BaseTrainer):
    
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
    
    def train(self):
        self.model.train()
        
        total_train_loss, n_correct, n_total = 0, 0, 0
        
        for batch_idx, (x, y) in enumerate(self.trainloader):
            x, y = x.to(self.device), y.to(self.device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=self.config.train.use_amp):
                outputs, _ = self.model(x)
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
                outputs, _ = self.model(x)
                loss = self.criterion(outputs, y)

                total_test_loss += loss.item()
                _, predicted = outputs.max(1)
                n_total += y.size(0)
                n_correct += predicted.eq(y).sum().item()
        
            avg_test_loss = total_test_loss / (batch_idx + 1)
            test_accuracy = 100. * n_correct / n_total
            print(f'Test Loss: {avg_test_loss:.3f} | Test Acc: {test_accuracy:.3f}% ({n_correct}/{n_total})')
        
        return total_test_loss, test_accuracy