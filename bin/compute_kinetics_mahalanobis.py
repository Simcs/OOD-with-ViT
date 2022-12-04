import json
import pickle
import torch
import json
import random
from tqdm import tqdm

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve

# random.seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def measure_performances(id_maha, ood_maha):
    results = [[] for _ in range(3)]
    for rn in tqdm(range(1000)):
        random.seed(rn)
        
        min_len = min(len(id_maha), len(ood_maha))
        sampled_id_maha = random.sample(id_maha, min_len)
        sampled_ood_maha = random.sample(ood_maha, min_len)

        test_y = [0 for _ in range(len(sampled_id_maha))] + [1 for _ in range(len(sampled_ood_maha))]
        ood_scores = sampled_id_maha + sampled_ood_maha
        # max_ood_score = max(ood_scores)
        # ood_scores = [i / max_ood_score for i in ood_scores]
        # print('# of k400 and k600 embeddings:', len(set(ood_scores)))

        fpr, tpr, auroc_score = auroc(test_y, ood_scores)
        pr, re, aupr_score = aupr(test_y, ood_scores)
        fpr95 = fpr_at_95_tpr(test_y, ood_scores)
        results[0].append(auroc_score)
        results[1].append(aupr_score)
        results[2].append(fpr95)
    
    import numpy as np
    aurocs, auprs, fpr95s = np.array(results[0]), np.array(results[1]), np.array(results[2])
    print('        max  min  mean  std')
    print(f'auroc: {aurocs.max():.6f}  {aurocs.min():.6f}  {aurocs.mean():.6f} {aurocs.std():.6f}')
    print(f'aupr: {auprs.max():.6f}  {auprs.min():.6f}  {auprs.mean():.6f} {auprs.std():.6f}')
    print(f'fpr95: {fpr95s.max():.6f}  {fpr95s.min():.6f}  {fpr95s.mean():.6f} {fpr95s.std():.6f}')
    print('auroc argmax', aurocs.argmax(), 'argmin', aurocs.argmin())
    print('auprs argmax', auprs.argmax(), 'argmin', auprs.argmin())
    print('fpr95s argmax', fpr95s.argmax(), 'argmin', fpr95s.argmin())

print('loading train statistics...')
with open('./data/kinetics/k400_train_statistics_total.pkl', 'rb') as f:
    train_stat_total = pickle.load(f)

# prepare statistics
means, precision = train_stat_total['mean'], train_stat_total['total_precision']
for k400_class in means:
    means[k400_class] = torch.Tensor(means[k400_class]).to(device)
precision = torch.Tensor(precision).to(device).float()

# kinetics400 vs. kinetics600
# compute k400 mahalanobis distances
print('loading k400 embeddings...')
k400_val = []
with open('./data/kinetics/embeddings/original/k400_val_embeddings.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        pre_logit = torch.Tensor(emb_js['penultimate'])
        k400_val.append(pre_logit.view(1, -1))
k400_val = torch.cat(k400_val, dim=0)

print('computing k400 mahalanobis distances...')
k400_mahalanobis_distances = []
for k400_features in k400_val.chunk(2, dim=0):
    k400_gaussian_scores = []
    k400_features = k400_features.to(device)
    for mean in tqdm(means.values()):
        zero_f = k400_features - mean
        gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
        k400_gaussian_scores.append(gau_term.cpu().view(-1, 1))
    k400_gaussian_scores = torch.cat(k400_gaussian_scores, dim=1)
    k400_maha, _ = k400_gaussian_scores.min(dim=1)
    k400_mahalanobis_distances.extend(k400_maha.numpy())
print('computing gaussian scores finished.')
# k400_mahalanobis_distances = k400_mahalanobis_distances / k400_mahalanobis_distances.max(dim=0)[0]
# k400_mahalanobis_distances = k400_mahalanobis_distances.numpy().tolist()
print('# of k400 embeddings:', len(k400_mahalanobis_distances))
    
# compute k600 mahalanobis distances
print('loading k600 embeddings...')
k600_original = []
with open('./data/kinetics/embeddings/original/k600_val_embeddings_deduplicated.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        pre_logit = torch.Tensor(emb_js['penultimate']).to(device)
        k600_original.append(pre_logit.view(1, -1))
k600_original = torch.cat(k600_original, dim=0)

print('computing k600 mahalanobis distances...')
k600_gaussian_scores = []
for mean in tqdm(means.values()):
    zero_f = k600_original - mean
    gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
    k600_gaussian_scores.append(gau_term.cpu().view(-1, 1))
print('computing gaussian scores finished.')
k600_gaussian_scores = torch.cat(k600_gaussian_scores, dim=1)
k600_mahalanobis_distances, _ = k600_gaussian_scores.min(dim=1)
# k600_mahalanobis_distances = k600_mahalanobis_distances / k600_mahalanobis_distances.max(dim=0)[0]
k600_mahalanobis_distances = k600_mahalanobis_distances.numpy().tolist()
print('# of k600 embeddings:', len(k600_mahalanobis_distances))

# min_len = min(len(k400_mahalanobis_distances), len(k600_mahalanobis_distances))
# k400_mahalanobis_distances = random.sample(k400_mahalanobis_distances, min_len)
# k600_mahalanobis_distances = random.sample(k600_mahalanobis_distances, min_len)

print('computing k400 vs. k600 ood scores...')
test_y = [0 for _ in range(len(k400_mahalanobis_distances))] + [1 for _ in range(len(k600_mahalanobis_distances))]
ood_scores = k400_mahalanobis_distances + k600_mahalanobis_distances
max_ood_score = max(ood_scores)
ood_scores = [i / max_ood_score for i in ood_scores]
print('# of k400 and k600 embeddings:', len(set(ood_scores)))

measure_performances(k400_mahalanobis_distances, k600_mahalanobis_distances)

fpr, tpr, k600_auroc_score = auroc(test_y, ood_scores)
pr, re, k600_aupr_score = aupr(test_y, ood_scores)
k600_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('auroc:', k600_auroc_score, 'aupr:', k600_aupr_score, 'fpr95:', k600_fpr95)
with open('./result/ood_scores/video/original/k400_vs_k600_maha.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': k600_auroc_score,
        'aupr': k600_aupr_score,
        'fpr95': k600_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/k400_vs_k600/maha_auroc.png')
save_precision_recall_curve(pr, re, './result/images/k400_vs_k600/maha_aupr.png')

# compute k700 mahalanobis distances
print('computing k700 mahalanobis distances...')
k700_original = []
with open('./data/kinetics/embeddings/original/k700-2020_val_embeddings_deduplicated.jsonl', 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        pre_logit = torch.Tensor(emb_js['penultimate']).to(device)
        k700_original.append(pre_logit.view(1, -1))
k700_original = torch.cat(k700_original, dim=0)

print('computing k700 mahalanobis distances...')
k700_gaussian_scores = []
for mean in tqdm(means.values()):
    zero_f = k700_original - mean
    gau_term = torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
    k700_gaussian_scores.append(gau_term.cpu().view(-1, 1))
print('computing gaussian scores finished.')
k700_gaussian_scores = torch.cat(k700_gaussian_scores, dim=1)
k700_mahalanobis_distances, _ = k700_gaussian_scores.min(dim=1)
# k700_mahalanobis_distances = k700_mahalanobis_distances / k700_mahalanobis_distances.max(dim=0)[0]
k700_mahalanobis_distances = k700_mahalanobis_distances.numpy().tolist()
print('# of k700 embeddings:', len(k700_mahalanobis_distances))

# min_len = min(len(k400_mahalanobis_distances), len(k700_mahalanobis_distances))
# k400_mahalanobis_distances = random.sample(k400_mahalanobis_distances, min_len)
# k700_mahalanobis_distances = random.sample(k700_mahalanobis_distances, min_len)

print('computing k400 vs. k700 ood scores...')
test_y = [0 for _ in range(len(k400_mahalanobis_distances))] + [1 for _ in range(len(k700_mahalanobis_distances))]
ood_scores = k400_mahalanobis_distances + k700_mahalanobis_distances
max_ood_score = max(ood_scores)
ood_scores = [i / max_ood_score for i in ood_scores]
print('# of k400 and k700 embeddings:', len(set(ood_scores)))

measure_performances(k400_mahalanobis_distances, k700_mahalanobis_distances)

fpr, tpr, k700_auroc_score = auroc(test_y, ood_scores)
pr, re, k700_aupr_score = aupr(test_y, ood_scores)
k700_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
print('k400 vs. k700:')
print('auroc:', k700_auroc_score, 'aupr:', k700_aupr_score, 'fpr95:', k700_fpr95)
with open('./result/ood_scores/video/original/k400_vs_k700_maha.jsonl', 'w') as f:
    f.write(json.dumps({
        'auroc': k700_auroc_score,
        'aupr': k700_aupr_score,
        'fpr95': k700_fpr95,
    }))
save_roc_curve(fpr, tpr, './result/images/k400_vs_k700/maha_auroc.png')
save_precision_recall_curve(pr, re, './result/images/k400_vs_k700/maha_aupr.png')
