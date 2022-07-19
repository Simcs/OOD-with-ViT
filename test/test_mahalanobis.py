import time
from pprint import pprint
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
# set_seed(1234)
# set_seed(4321)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_root = Path('./logs')
checkpoint_path = log_root / 'checkpoints'

# create ConfigDict from config yaml file
import yaml
from ml_collections import config_dict

config_path = Path('configs') / 'ViT-OOD-CIFAR10.yaml'
with config_path.open('r') as f:
    config = yaml.safe_load(f)
    config = config_dict.ConfigDict(config)
    
# initialize ViT model and load pretrained weights
from models.vit import ViT

model = ViT(
    image_size=config.dataset.img_size,
    patch_size=config.model.patch_size,
    num_classes=len(config.dataset.in_distribution_class_indices),
    dim=config.model.dim_head,
    depth=config.model.depth,
    heads=config.model.n_heads,
    mlp_dim=config.model.dim_mlp,
    dropout=config.model.dropout,
    emb_dropout=config.model.emb_dropout,
    visualize=True,
)

model = model.to(device=device)
# print(model)

model_name = config.model.name
patch_size = config.model.patch_size
checkpoint = torch.load(checkpoint_path / f'{model_name}-patch{patch_size}-ckpt.t7')

state_dict = checkpoint['model_state_dict']
trimmed_keys = []
for key in state_dict.keys():
    # remove prefix 'module.' for each key (in case of DataParallel)
    trimmed_keys.append(key[7:])
trimmed_state_dict = OrderedDict(list(zip(trimmed_keys, state_dict.values())))

model.load_state_dict(trimmed_state_dict)

from datasets import OOD_CIFAR10

dataset_mean, dataset_std = config.dataset.mean, config.dataset.std
dataset_root = config.dataset.root
img_size = config.dataset.img_size
in_distribution_class_indices = config.dataset.in_distribution_class_indices

transform_test = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std),
])      

dataset = OOD_CIFAR10(
    root=dataset_root, 
    in_distribution_class_indices=in_distribution_class_indices, 
    train=False, 
    download=False, 
    transform=transform_test
)

class_indices = list(range(10))
idx_to_class = dict((v, k) for k, v in dataset.class_to_idx.items())
ood_class_indices = [i for i in class_indices if i not in in_distribution_class_indices]
id_class_indices = [i for i in class_indices if i in in_distribution_class_indices]
print(dataset.class_to_idx)

from sklearn.metrics import roc_auc_score, roc_curve

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
    
from metrics.mahalanobis import Mahalanobis

test_y, ood_score = [], []
msp = Mahalanobis(config, model)
print('processing in-distribution samples...')
for id_idx in id_class_indices:
    zero_f_sum = 0.
    for img in tqdm(dataset.img_dict[id_idx][:100]):
        test_y.append(0)
        score, zero_f_mean = msp.compute_ood_score(img)
        zero_f_sum += zero_f_mean
        ood_score.append(score)
        # ood_score.append(msp.compute_ood_score(img))
        # print(msp.compute_ood_score(img))
    print('zero f mean:', zero_f_sum / (len(id_class_indices) * 100))

print('processing out-of-distribution samples...')   
for ood_idx in ood_class_indices:
    zero_f_sum = 0. 
    for img in tqdm(dataset.img_dict[ood_idx]):
        test_y.append(1)
        score, zero_f_mean = msp.compute_ood_score(img)
        zero_f_sum += zero_f_mean
        ood_score.append(score)
        # ood_score.append(msp.compute_ood_score(img))
    print('zero f mean:', zero_f_sum / (len(id_class_indices) * 1000))
        
fper, tper, threshold = roc_curve(test_y, ood_score)
auroc_score = roc_auc_score(test_y, ood_score)

plot_roc_curve(fper, tper)
print('Mahalanobis AUROC score:', auroc_score)