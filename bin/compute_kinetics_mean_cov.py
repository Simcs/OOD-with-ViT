import json
import pickle
import os
import torch
import numpy as np
from tqdm import tqdm

from torchvision.datasets import Kinetics
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance


correct_penultimates = {}
# wrong_penultimates = {}

filename = './data/kinetics/embeddings/original/k400_train_embeddings.jsonl'
with open(filename, 'r') as f:
    for line in tqdm(f):
        emb_js = json.loads(line)
        penultimate_feature = torch.Tensor(emb_js['penultimate'])
        gt_label, pred_label = emb_js['gt'], emb_js['pred']
        if gt_label not in correct_penultimates:
            correct_penultimates[gt_label] = []
        # if gt_label not in wrong_penultimates:
        #     wrong_penultimates[gt_label] = []
            
        # if gt_label == pred_label:
        correct_penultimates[gt_label].append(penultimate_feature.view(1, -1))
        # else:
        #     wrong_penultimates[gt_label].append(penultimate_feature)
    

correct_means = {c: None for c in correct_penultimates.keys()}
for c in correct_penultimates.keys():
    correct_penultimates[c] = torch.cat(correct_penultimates[c], dim=0)
    correct_means[c] = torch.mean(correct_penultimates[c], dim=0)


# total
group_lasso = EmpiricalCovariance(assume_centered=False)
X = []
for c in correct_penultimates.keys():
    X.append(correct_penultimates[c] - correct_means[c])
X = torch.cat(X, dim=0).numpy()
group_lasso.fit(X)
total_precision = torch.from_numpy(group_lasso.precision_).float()
print('covariance norm:', np.linalg.norm(group_lasso.precision_))
result_total = {
    'mean': correct_means,
    'total_precision': total_precision.numpy().tolist(),
}

# classwise
classwise_precisions = {}
for c in correct_penultimates.keys():
    group_lasso = ShrunkCovariance(assume_centered=False)
    X = (correct_penultimates[c] - correct_means[c]).numpy()
    group_lasso.fit(X)
    precision = torch.from_numpy(group_lasso.precision_).float()
    classwise_precisions[c] = precision
norms = [np.linalg.norm(classwise_precisions[c]) for c in classwise_precisions.keys()]
print('covairance norms:', norms)
result_classwise = {}
for c in correct_means.keys():
    result_classwise[c] = {
        'mean': correct_means[c].numpy().tolist(),
        'classwise_precision': classwise_precisions[c].tolist(),
    }


with open('./data/kinetics/k400_train_statistics_total.pkl', 'wb') as f:
    pickle.dump(result_total, f)

with open('./data/kinetics/k400_train_statistics_classwise.pkl', 'wb') as f:
    pickle.dump(result_classwise, f)