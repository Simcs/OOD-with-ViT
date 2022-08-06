import time
from pprint import pprint
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import dash_html_components as html
import dash_core_components as dcc
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output, no_update

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import argparse


# create ConfigDict from config yaml file
import yaml
from ml_collections import config_dict

config_path = Path('configs') / 'deit_tiny-pretrained-cifar10.yaml'
with config_path.open('r') as f:
    config = yaml.safe_load(f)
    config = config_dict.ConfigDict(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from dash import no_update
import plotly.graph_objects as go
import pandas as pd

# initialize ViT model and load pretrained weights
from ood_with_vit.models.vit import ViT

# frequently used variables
model_name = config.model.name
patch_size = config.model.patch_size
summary = config.summary

# log directories
log_root = Path('./logs') / model_name / summary
checkpoint_path = log_root / 'checkpoints'

def initialize_vit_model(config, verbose=0):
    n_class = config.dataset.n_class
    if config.model.pretrained:
        model = torch.hub.load(
            repo_or_dir=config.model.repo,
            model=config.model.pretrained_model,
            pretrained=False,
        )
        model.head = nn.Linear(model.head.in_features, n_class)
    else:
        model = ViT(
            image_size=config.model.img_size,
            patch_size=config.model.patch_size,
            num_classes=n_class,
            dim=config.model.dim_head,
            depth=config.model.depth,
            heads=config.model.n_heads,
            mlp_dim=config.model.dim_mlp,
            dropout=config.model.dropout,
            emb_dropout=config.model.emb_dropout,
            visualize=True,
        )

    model = model.to(device=device)
    if verbose:
        print(model)

    checkpoint = torch.load(checkpoint_path / f'{summary}_best.pt')

    state_dict = checkpoint['model_state_dict']
    trimmed_keys = []
    for key in state_dict.keys():
        # remove prefix 'module.' for each key (in case of DataParallel)
        trimmed_keys.append(key[7:])
    trimmed_state_dict = OrderedDict(list(zip(trimmed_keys, state_dict.values())))

    model.load_state_dict(trimmed_state_dict)
    return model

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset

dataset_mean, dataset_std = config.dataset.mean, config.dataset.std
dataset_root = config.dataset.root
img_size = config.model.img_size

transform_test = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(dataset_mean, dataset_std),
])      

cifar10 = CIFAR10(
    root=dataset_root, 
    train=False, 
    download=False, 
    transform=transform_test
)
id_test_dataloader = DataLoader(
    dataset=cifar10, 
    batch_size=config.eval.batch_size, 
    shuffle=False, 
    num_workers=8
)

cifar100 = CIFAR100(
    root=dataset_root, 
    train=False, 
    download=False, 
    transform=transform_test
)
ood_test_dataloader = DataLoader(
    dataset=cifar100, 
    batch_size=config.eval.batch_size, 
    shuffle=False, 
    num_workers=8
)

# set seeds
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 1234

from ood_with_vit.visualizer.feature_extractor import FeatureExtractor

model = initialize_vit_model(config)
# add hooks for feature extraction
feature_extractor = FeatureExtractor(
    model=model,
    layer_name=config.model.layer_name.penultimate,
)

# gather penultimate features
from ood_with_vit.utils import compute_penultimate_features

set_seed(seed)
num_samples = 20
num_class = 10

print('processing in-distribution samples...')
X_in, y_in = [], []
in_img_paths = []
cls_indices = random.sample(range(len(cifar10.classes)), num_class)
for cls_idx in cls_indices:
    img_indices = np.where(np.array(cifar10.targets) == cls_idx)[0]
    img_indices = random.sample(list(img_indices), num_samples)
    for i in tqdm(img_indices):
        img, _ = cifar10[i]
        img = img.to(device)
        penultimate_features = compute_penultimate_features(
            config=config, 
            model=model, 
            imgs=img.unsqueeze(0),
            feature_extractor=feature_extractor,    
        )
        X_in.append(penultimate_features.squeeze().numpy())
        y_in.append((cls_idx, cifar10.classes[cls_idx]))
        img_path = f'/assets/cifar10/test/{i:05d}.jpg'
        in_img_paths.append(img_path)
        
print('processing out-of-distribution samples...')
X_out, y_out = [], []
out_img_paths = []
cls_indices = random.sample(range(len(cifar100.classes)), num_class)
for cls_idx in cls_indices:
    img_indices = np.where(np.array(cifar100.targets) == cls_idx)[0]
    img_indices = random.sample(list(img_indices), num_samples)
    for i in tqdm(img_indices):
        img, _ = cifar100[i]
        img = img.to(device)
        penultimate_features = compute_penultimate_features(
            config=config, 
            model=model, 
            imgs=img.unsqueeze(0),
            feature_extractor=feature_extractor,    
        )
        X_out.append(penultimate_features.squeeze().numpy())
        y_out.append((cls_idx, cifar100.classes[cls_idx]))
        img_path = f'/assets/cifar100/test/{i:05d}.jpg'
        out_img_paths.append(img_path)


from sklearn.manifold import TSNE

def visualize_results(X, y, img_paths, seed, port):

    tsne = TSNE(n_components=2, random_state=seed)
    X_tsne = tsne.fit_transform(X)

    df_embed = pd.DataFrame(X_tsne)
    df_embed = df_embed.rename(columns={0: 'x', 1: 'y'})
    df_embed = df_embed.assign(label=y)
    df_embed = df_embed.assign(img=img_paths)

    partitioned_info, processed_labels = [], []
    for label in y:
        if label not in processed_labels:
            processed_labels.append(label)
            partitioned_info.append(df_embed.where(df_embed['label'] == label).dropna())
            
    fig = px.scatter(
        df_embed,
        x='x',
        y='y',
        color='label',
        labels={'label': 'class'},
        title='ImageNet pretrained, CIFAR10 finetuned ViT CIFAR10 T-SNE',
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    app = JupyterDash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        curveNum = pt["curveNumber"]

        df_row = partitioned_info[curveNum].iloc[num]

        img = df_row['img']
        label = df_row['label']

        children = [
            html.Div([
                html.Img(src=img, style={'width': '70%'}),
                html.P(f"{img}", style={'font-size': '0.7rem', 'padding-bottom': '8px'}),
                html.P([html.B('Label: '), f'{label[1]}']),
            ], style={'width': '200px', 'white-space': 'normal', 'text-align': 'center', 'line-height': '0'})
        ]

        return True, bbox, children

    app.run_server(debug=True, use_reloader=False, port=port)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', type=str, help='distribution type')
    args = parser.parse_args()

    assert args.dist in ['in', 'out']

    seed = 1234
    port = 38051
    if args.dist == 'in':
        visualize_results(X_in, y_in, in_img_paths, seed, port)
    else:
        visualize_results(X_out, y_out, out_img_paths, seed, port)