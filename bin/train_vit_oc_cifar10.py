import os
import argparse
import csv
import yaml
import time
from pathlib import Path

import wandb
from ml_collections import config_dict

import torch

from trainer import ViT_OC_CIFAR10_Trainer


def main(config):
    
    # create log directories
    log_root = Path('./logs')
    checkpoint_path = log_root / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # frequently used variables
    model_name = config.model.name
    base_lr = config.optimizer.base_lr
    batch_size = config.train.batch_size
    n_indist_class = len(config.dataset.in_distribution_class_indices)
    patch_size = config.model.patch_size
    
    # init wandb
    name = f"{model_name}_OC{n_indist_class}_lr{base_lr}_bs{batch_size}"
    wandb.init(
        project=f"ViT-OC{n_indist_class}-CIFAR10",
        name=name,
        config=config,
    )
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if config.train.scheduler == 'cosine':
        wandb.config.scheduler = "cosine"
    else:
        wandb.config.scheduler = "ReduceLROnPlateau"

    losses = []
    accs = []

    wandb.watch(trainer.model)
    for epoch in range(start_epoch, config.train.n_epochs):
        print(f'\nEpoch: {epoch}')
        start = time.time()
        train_loss = trainer.train()
        val_loss, val_acc = trainer.test()
        trainer.step_scheduler(val_loss)
        
        # Save checkpoint.
        if val_acc > best_acc:
            print('Saving..')
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scaler_state_dict': trainer.scaler.state_dict(),
            }
            torch.save(state, f'./checkpoint/{model_name}-patch{patch_size}-ckpt.t7')
            best_acc = val_acc
        
        content = time.ctime() \
            + f' Epoch {epoch}, lr: {trainer.optimizer.param_groups[0]["lr"]:.7f}' \
            + f' val loss: {val_loss:.5f}, acc: {(val_acc):.5f}, time: {time.time() - start:.3f}'
        print(content)
        log_file = log_root / f'log_{model_name}_patch{patch_size}.txt'
        with log_file.open(mode='a') as f:
            f.write(content + "\n")
        
        losses.append(val_loss)
        accs.append(val_acc)
        
        # Log training..
        wandb.log({
            'epoch': epoch, 
            'train_loss': train_loss, 
            'val_loss': val_loss, 
            'val_acc': val_acc, 
            'lr': trainer.optimizer.param_groups[0]['lr'],
            "epoch_time": time.time() - start
        })

        # Write out csv..
        # with open(f'log/log_{model_name}_patch{patch_size}.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(losses) 
        #     writer.writerow(accs) 
        # print(losses)

    # writeout wandb
    wandb.save(f"wandb_{model_name}.h5")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--config', required=True, type=str, help='config filename')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file) 
        config = config_dict.ConfigDict(config)
    
    trainer = ViT_OC_CIFAR10_Trainer(config)
        
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        model_name = config.model.name
        patch_size = config.model.patch_size
        if model_name == 'vit':
            checkpoint = torch.load(f'./checkpoint/{model_name}-patch{patch_size}-ckpt.t7')
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    main(config)


