import os
import argparse
import yaml
import time
from pathlib import Path

import wandb
from ml_collections import config_dict

import torch

from ood_with_vit.trainer import (
    ViT_OOD_CIFAR10_Trainer, 
    ViT_Finetune_OOD_CIFAR10_Trainer,
)

def create_trainer(config):
    if config.model.pretrained:
        trainer = ViT_Finetune_OOD_CIFAR10_Trainer(config)
    else:
        trainer = ViT_OOD_CIFAR10_Trainer(config)
    return trainer

def main(config, args):
    
    trainer = create_trainer(config)
    
    # frequently used variables
    model_name = config.model.name
    base_lr = config.optimizer.base_lr
    batch_size = config.train.batch_size
    patch_size = config.model.patch_size
    log_epoch = config.train.log_epoch
    summary = config.summary
    
    # create log directories
    log_root = Path('./logs') / model_name
    checkpoint_path = log_root / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # init wandb
    name = f"{summary}_lr{base_lr}_bs{batch_size}"
    wandb.init(
        project=f"OOD-with-ViT",
        name=name,
        config=config,
    )
    if config.train.scheduler == 'cosine':
        wandb.config.scheduler = "cosine"
    else:
        wandb.config.scheduler = "ReduceLROnPlateau"
    
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    if args.resume:
        # load checkpoint
        print('resuming from checkpoint...')
        assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
        
        checkpoint = torch.load(checkpoint_path / args.checkpoint)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    wandb.watch(trainer.model)
    for epoch in range(start_epoch, config.train.n_epochs):
        print(f'\nEpoch: {epoch}')
        start = time.time()
        train_loss = trainer.train()
        val_loss, val_acc = trainer.test()
        trainer.step_scheduler(val_loss)
        
        # save checkpoint per every log epoch.
        if epoch % log_epoch == 0:
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scaler_state_dict': trainer.scaler.state_dict(),
            }
            torch.save(state, checkpoint_path / f'{summary}_{epoch}.pt')
            
        # save checkpoint if best accuracy achieved.
        if val_acc > best_acc:
            print('Save checkpoint...')
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scaler_state_dict': trainer.scaler.state_dict(),
            }
            torch.save(state, checkpoint_path / f'{summary}_best.pt')
        
        # log training info
        content = time.ctime() \
            + f' Epoch {epoch}, lr: {trainer.optimizer.param_groups[0]["lr"]:.7f}' \
            + f' val loss: {val_loss:.5f}, acc: {(val_acc):.5f}, time: {time.time() - start:.3f}'
        log_file = log_root / f'{summary}.txt'
        with log_file.open(mode='a') as f:
            f.write(content + "\n")
        print(content)
        
        # log to wandb
        wandb.log({
            'epoch': epoch, 
            'train_loss': train_loss, 
            'val_loss': val_loss, 
            'val_acc': val_acc, 
            'lr': trainer.optimizer.param_groups[0]['lr'],
            "epoch_time": time.time() - start
        })

    # writeout wandb
    wandb.save(f"wandb_{summary}.h5")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--config', required=True, type=str, help='config filename')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path to resume from')
    args = parser.parse_args()
    
    # load yaml config and converts to ConfigDict
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file) 
        config = config_dict.ConfigDict(config)
          
    main(config, args)


