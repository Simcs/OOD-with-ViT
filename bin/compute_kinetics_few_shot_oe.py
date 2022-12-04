import time
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse

from ood_with_vit.exposure.kinetics_ood import KineticsOOD
from ood_with_vit.exposure.outlier_exposure import OutlierExposure
from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
    

def compute_few_shot_oe_ood_scores(id, ood, args):
    
    n_shot = args.n_shot
    device = 'cuda'
    
    kinetics_oe_train = KineticsOOD(
        mode='original',
        id=id,
        ood=ood,
        train=True,
        n_shot=n_shot,
    )
    kinetics_oe_train_dl = DataLoader(
        dataset=kinetics_oe_train,
        batch_size=1024,
        shuffle=True,
        num_workers=8,
    )

    kinetics_oe_val = KineticsOOD(
        mode='original',
        id=id,
        ood=ood,
        train=False,
        _id_embeddings=kinetics_oe_train.id_embeddings,
        _id_labels=kinetics_oe_train.id_labels,
        _ood_embeddings=kinetics_oe_train.ood_embeddings,
        _ood_labels=kinetics_oe_train.ood_labels,
    )
    kinetics_oe_val_dl = DataLoader(
        dataset=kinetics_oe_val,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
    )

    model = OutlierExposure(
        id=id,
        ood=ood
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # train
    for epoch in range(50):
        total_train_loss, n_correct, n_total = 0, 0, 0
        model.train()
        for batch_idx, (x, y) in enumerate(kinetics_oe_train_dl):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            n_total += y.size(0)
            n_correct += predicted.eq(y).sum().item()
        
        avg_train_loss = total_train_loss / (batch_idx + 1)
        train_accuracy = 100. * n_correct / n_total
        if epoch % 10 == 0:
            print(f'epoch {epoch}, train loss: {avg_train_loss:.3f} | train acc: {train_accuracy:.3f} ({n_correct}/{n_total})')
        
        if train_accuracy > 98.:
            break
    
    print(f'epoch {epoch}, train loss: {avg_train_loss:.3f} | train acc: {train_accuracy:.3f} ({n_correct}/{n_total})')
    
    # eval
    model.eval()
    total_test_loss, n_correct, n_total = 0, 0, 0
    ood_scores, ood_gt = [], []
    softmax = nn.Softmax(dim=0)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(kinetics_oe_val_dl):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_test_loss += loss.item()
            _, predicted = outputs.max(1)
            n_total += y.size(0)
            n_correct += predicted.eq(y).sum().item()
            
            for output, gt in zip(outputs, y):
                # print('output:', output.shape, 'gt:', gt.shape, gt.item())
                # print('ood score:', softmax(output)[1].item())
                ood_scores.append(softmax(output)[400:].sum().item())
                if gt.item() >= 400:
                    ood_gt.append(1)
                else:
                    ood_gt.append(0)

        _, _, auroc_score = auroc(ood_gt, ood_scores)
        _, _, aupr_score = aupr(ood_gt, ood_scores)
        fpr95 = fpr_at_95_tpr(ood_gt, ood_scores)
        
        avg_test_loss = total_test_loss / (batch_idx + 1)
        test_accuracy = 100. * n_correct / n_total
        print(f'Test Loss: {avg_test_loss:.3f} | Test Acc: {test_accuracy:.3f}% ({n_correct}/{n_total})')
        print(f'{id}vs{ood} auroc:', auroc_score, 'aupr:', aupr_score, 'fpr95:', fpr95)
    
    
    return auroc_score, aupr_score, fpr95
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument('--n_shot', type=int)
    args = parser.parse_args()
    
    # k400 vs. k600
    # auroc_score, aupr_score, fpr95 = compute_few_shot_oe_ood_scores('k400', 'k600', args)
    aurocs, auprs, fpr95s = [], [], []
    for _ in range(10):
        auroc_score, aupr_score, fpr95 = compute_few_shot_oe_ood_scores('k400', 'k600', args)
        aurocs.append(auroc_score)
        auprs.append(aupr_score)
        fpr95s.append(fpr95)
    mean_auroc, std_auroc = np.mean(aurocs), np.std(aurocs)
    mean_aupr, std_aupr = np.mean(auprs), np.std(auprs)
    mean_fpr95, std_fpr95 = np.mean(fpr95s), np.std(fpr95s)
        
    result_filename = Path('./result') / 'ood_scores' / 'video' / 'original' / 'k400_vs_k600.jsonl'
    result_filename = result_filename.parent / (result_filename.stem + f'_oe_{args.n_shot}' + result_filename.suffix)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'mean_aupr': mean_aupr,
            'std_aupr': std_aupr,
            'mean_fpr95': mean_fpr95,
            'std_fpr95': std_fpr95,
        }))
    
    # k400 vs. k700
    # auroc_score, aupr_score, fpr95 = compute_few_shot_oe_ood_scores('k400', 'k700-2020', args)
    aurocs, auprs, fpr95s = [], [], []
    for _ in range(10):
        auroc_score, aupr_score, fpr95 = compute_few_shot_oe_ood_scores('k400', 'k700-2020', args)
        aurocs.append(auroc_score)
        auprs.append(aupr_score)
        fpr95s.append(fpr95)
    mean_auroc, std_auroc = np.mean(aurocs), np.std(aurocs)
    mean_aupr, std_aupr = np.mean(auprs), np.std(auprs)
    mean_fpr95, std_fpr95 = np.mean(fpr95s), np.std(fpr95s)
    
    result_filename = Path('./result') / 'ood_scores' / 'video' / 'original' / 'k400_vs_k700.jsonl'
    result_filename = result_filename.parent / (result_filename.stem + f'_oe_{args.n_shot}' + result_filename.suffix)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'mean_auroc': mean_auroc,
            'std_auroc': std_auroc,
            'mean_aupr': mean_aupr,
            'std_aupr': std_aupr,
            'mean_fpr95': mean_fpr95,
            'std_fpr95': std_fpr95,
        }))