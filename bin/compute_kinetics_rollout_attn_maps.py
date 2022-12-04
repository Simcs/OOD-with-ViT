import os
import time
from pprint import pprint
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import argparse

from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import video_transformer.data_transform as T
from video_transformer.transformer import ClassificationHead

from video_transformer.video_transformer import ViViT
from ood_with_vit.datasets.kinetics import VideoOnlyKinetics
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor
from ood_with_vit.visualizer.attention_rollout import ViTAttentionRollout


def replace_state_dict(state_dict):
	for old_key in list(state_dict.keys()):
		if old_key.startswith('model'):
			new_key = old_key[6:]
			state_dict[new_key] = state_dict.pop(old_key)
		else:
			new_key = old_key[9:]
			state_dict[new_key] = state_dict.pop(old_key)

def init_from_pretrain_(module, pretrained, init_module):
    if torch.cuda.is_available():
        state_dict = torch.load(pretrained)
    else:
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
    if init_module == 'transformer':
        replace_state_dict(state_dict)
    elif init_module == 'cls_head':
        replace_state_dict(state_dict)
    else:
        raise TypeError(f'pretrained weights do not include the {init_module} module')
    msg = module.load_state_dict(state_dict, strict=False)
    return msg


def create_model():
    num_frames = 8
    frame_interval = 32
    num_class = 400
    arch = 'vivit' # turn to vivit for initializing vivit model

    pretrain_pth = './logs/vivit/vivit_model.pth'
    num_frames = num_frames * 2
    frame_interval = frame_interval // 2
    model = ViViT(
        num_frames=num_frames,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        attention_type='fact_encoder',
        return_cls_token=True,
        pretrain_pth=pretrain_pth,
        weights_from='kinetics',
    )

    cls_head = ClassificationHead(num_classes=num_class, in_channels=768)
    # msg_trans = init_from_pretrain_(model, pretrain_pth, init_module='transformer')
    msg_cls = init_from_pretrain_(cls_head, pretrain_pth, init_module='cls_head')
    print(f'load model finished, the missing key of cls is:{msg_cls[0]}')
    
    return model, cls_head


def compute_rollout_attention_maps(args):
    dataset, split, force = args.dataset, args.split, args.force
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    print(f'computing {dataset} split {split}...')
    
    attn_map_filename = f'./data/kinetics/{dataset}_{split}_{head_fusion}_{discard_ratio}_attn_maps.jsonl'
    # embeddings_filename = os.path.expanduser(embeddings_filename)
    
    # if not force and os.path.exists(attn_map_filename):
    #     return
    
    # prepare metadata
    print('preparing metadata...')
    # with open(f'./data/kinetics/{dataset}_{split}_metadata.pkl', 'rb') as f:
    #     metadata = pickle.load(f)
    with open(f'./data/kinetics/k400_val_metadata.pkl', 'rb') as f:
        k400_metadata = pickle.load(f)
    print('preparing metadata complete')
    
    # prepare datasets and dataloaders
    dataset_root = f'~/workspace/dataset/kinetics/{dataset}'
    dataset_root = os.path.expanduser(dataset_root)
    
    k400_root = f'~/workspace/dataset/kinetics/k400'
    k400_root = os.path.expanduser(k400_root)
    
    dataset_mean, dataset_std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
    val_transform = T.create_video_transform(
        input_size=224,
        is_training=False,
        interpolation='bicubic',
        mean=dataset_mean,
        std=dataset_std,
    )
    
    print('preparing dataset...')
    kinetics_ds = VideoOnlyKinetics(
        root=dataset_root,
        frames_per_clip=16,
        split=split,
        num_workers=8,
        frame_rate=2,
        step_between_clips=1,
        transform=val_transform,
        # _precomputed_metadata=metadata,  
    )
    print('preparing dataset complete')
    
    k400_ds = VideoOnlyKinetics(
        root=k400_root,
        frames_per_clip=16,
        split='val',
        num_workers=8,
        frame_rate=2,
        step_between_clips=1,
        transform=val_transform,
        _precomputed_metadata=k400_metadata,  
    )

    kinetics_dl = DataLoader(
        dataset=kinetics_ds,
        batch_size=32,
        shuffle=False,
        num_workers=16,
    )
    
    # prepare models
    model, cls_head = create_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cls_head = model.to(device), cls_head.to(device)
    
    rollout = ViTAttentionRollout(
        head_fusion=head_fusion,
        discard_ratio=discard_ratio,
    )
    attention_extractor = FeatureExtractor(
        model=model,
        layer_name='attn_drop',
    )
    attention_extractor.hook()
    
    # compute embeddings
    n_correct, n_total = 0, 0
    id, cache_rate = 0, 100
    with open(attn_map_filename, 'r') as fp:
        lines = fp.readlines()
        attn_map = json.loads(lines[-1])
        id = attn_map['id'] + 1
        print(id)
        input()
    logs = []

    model.eval()
    cls_head.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(kinetics_dl)):
            if batch_idx % cache_rate == 0:
                with open(attn_map_filename, 'a+') as f:
                    for log in logs:
                        f.write(json.dumps(log) + '\n')
                logs = []
                
            x, y = x.to(device), y.to(device)
            pre_logits = attention_extractor(x)
            logits = cls_head(pre_logits)
            _, predicted = logits.max(1)
            
            spatial_attn_maps = attention_extractor.features[:12]
            rollout_spatial_attn_maps = rollout.spatial_rollout(spatial_attn_maps)
            rollout_spatial_attn_maps = rearrange(rollout_spatial_attn_maps, '(b t) p d -> b t p d', b=x.size(0))
            
            temporal_attn_maps = attention_extractor.features[12:]
            rollout_temporal_attn_maps = rollout.temporal_rollout(temporal_attn_maps)
            rollout_temporal_attn_maps = np.expand_dims(rollout_temporal_attn_maps, axis=1)
            
            # print(type(rollout_spatial_attn_maps), rollout_spatial_attn_maps.shape)
            # print(type(rollout_temporal_attn_maps), rollout_temporal_attn_maps.shape)
            
            for rollout_spatial_attn_map, rollout_temporal_attn_map, gt, pred \
                in zip(rollout_spatial_attn_maps, rollout_temporal_attn_maps, y, predicted):
                rollout_spatial_attn_map = rollout_spatial_attn_map.tolist()
                rollout_temporal_attn_map = rollout_temporal_attn_map.tolist()               
                gt_label = kinetics_ds.classes[gt.item()]
                pred_label = k400_ds.classes[pred.item()]
                logs.append({
                    'id': id, 
                    'gt': gt_label, 
                    'pred': pred_label,
                    'spatial_attention_map': rollout_spatial_attn_map,
                    'temporal_attention_map': rollout_temporal_attn_map,                
                })
                id += 1

            n_total += y.size(0)
            n_correct += predicted.eq(y).sum().item()

        with open(attn_map_filename, 'a+') as f:
            for log in logs:
                f.write(json.dumps(log) + '\n')
        logs = []
        
        test_accuracy = 100. * n_correct / n_total
        print(f'Test Acc: {test_accuracy:.3f}% ({n_correct}/{n_total})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['k400', 'k600', 'k700-2020'])
    parser.add_argument('--split', choices=['train', 'val'])
    parser.add_argument('--head_fusion', default='max')
    parser.add_argument('--discard_ratio', default=0.5)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    compute_rollout_attention_maps(args)