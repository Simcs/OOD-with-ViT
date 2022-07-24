from typing import Tuple

from ml_collections.config_dict import ConfigDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import transformers

class BaseTrainer:
    
    def __init__(self, config: ConfigDict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._create_model()
        self.trainloader, self.testloader = self._create_dataloader()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=self.config.train.use_amp)
        self.criterion = nn.CrossEntropyLoss()
    
    def _create_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError()
            
    def _create_model(self) -> nn.Module:
        raise NotImplementedError() 
    
    def _create_optimizer(self) -> Optimizer:
        optimizer_name = self.config.optimizer.name
        lr = self.config.optimizer.base_lr
        weight_decay = self.config.optimizer.weight_decay
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    
    def _create_scheduler(self):
        # use cosine or reduce LR on Plateau scheduling
        scheduler_name = self.config.scheduler.name
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, 
                T_max=self.config.train.n_epochs
            )
        elif scheduler_name == 'cosine_with_hard_restarts_with_warmup':
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.config.scheduler.warmup_steps,
                num_training_steps=self.config.scheduler.num_training_steps,
                num_cycles=self.config.scheduler.num_cycles,
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
        scheduler_name = self.config.scheduler.name
        if scheduler_name == 'cosine':
            self.scheduler.step()
        elif scheduler_name == 'cosine_with_hard_restarts_with_warmup':
            self.scheduler.step()
        else:
            self.scheduler.step(validation_loss)

