import json
import pickle
import torch
from tqdm import tqdm
import random

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve

random.seed(1234)

# kinetics400 vs. kinetics600
# compute k400 mahalanobis distances
print('loading k400 embeddings...')
k400_val = []
with open('./data/kinetics/embeddings/original/k400_val_embeddings.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        k400_val.append(logit.view(1, -1))
k400_val = torch.cat(k400_val, dim=0)
k400_msp, _ = k400_val.max(dim=1)
k400_msp = (-k400_msp.numpy()).tolist()
    
# compute k600 mahalanobis distances
print('loading k600 embeddings...')
k600_original = []
with open('./data/kinetics/embeddings/original/k600_val_embeddings_deduplicated.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        k600_original.append(logit.view(1, -1))
k600_original = torch.cat(k600_original, dim=0)
k600_msp, _ = k600_original.max(dim=1)
k600_msp = (-k600_msp.numpy()).tolist()

min_len = min(len(k400_msp), len(k600_msp))
k400_msp = random.sample(k400_msp, min_len)
k600_msp = random.sample(k600_msp, min_len)

print('computing k400 vs. k600 ood scores...')
test_y = [0 for _ in range(len(k400_msp))] + [1 for _ in range(len(k600_msp))]
ood_scores = k400_msp + k600_msp

fpr, tpr, k600_auroc_score = auroc(test_y, ood_scores)
pr, re, k600_aupr_score = aupr(test_y, ood_scores)
k600_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('auroc:', k600_auroc_score, 'aupr:', k600_aupr_score, 'fpr95:', k600_fpr95)
with open('./result/ood_scores/video/original/k400_vs_k600_msp.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': k600_auroc_score,
        'aupr': k600_aupr_score,
        'fpr95': k600_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/k400_vs_k600/msp_auroc.png')
save_precision_recall_curve(pr, re, './result/images/k400_vs_k600/msp_aupr.png')

# compute k700 mahalanobis distances
print('computing k700 mahalanobis distances...')
k700_original = []
with open('./data/kinetics/embeddings/original/k700-2020_val_embeddings_deduplicated.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        logit = torch.Tensor(emb_js['logit'])
        k700_original.append(logit.view(1, -1))
k700_original = torch.cat(k700_original, dim=0)
k700_msp, _ = k700_original.max(dim=1)
k700_msp = (-k700_msp.numpy()).tolist()

min_len = min(len(k400_msp), len(k700_msp))
k400_msp = random.sample(k400_msp, min_len)
k700_msp = random.sample(k700_msp, min_len)

print('computing k400 vs. k700 ood scores...')
test_y = [0 for _ in range(len(k400_msp))] + [1 for _ in range(len(k700_msp))]
ood_scores = k400_msp + k700_msp

fpr, tpr, k700_auroc_score = auroc(test_y, ood_scores)
pr, re, k700_aupr_score = aupr(test_y, ood_scores)
k700_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('k400 vs. k700:')
print('auroc:', k700_auroc_score, 'aupr:', k700_aupr_score, 'fpr95:', k700_fpr95)
with open('./result/ood_scores/video/original/k400_vs_k700_msp.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': k700_auroc_score,
        'aupr': k700_aupr_score,
        'fpr95': k700_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/k400_vs_k700/msp_auroc.png')
save_precision_recall_curve(pr, re, './result/images/k400_vs_k700/msp_aupr.png')