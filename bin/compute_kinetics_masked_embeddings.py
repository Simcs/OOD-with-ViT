import os
import time
from pprint import pprint
from pathlib import Path
from venv import create
from tqdm import tqdm
import pickle
import json
import argparse
import numpy as np

import yaml
from ml_collections import config_dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import video_transformer.data_transform as T
from video_transformer.transformer import ClassificationHead

from video_transformer.video_transformer import ViViT
from ood_with_vit.datasets.kinetics import VideoOnlyKinetics

from ood_with_vit.visualizer.feature_extractor import FeatureExtractor
from ood_with_vit.mim.video_spatial_attention_masking import VideoSpatialAttentionMaskingHooker
from ood_with_vit.mim.video_temporal_attention_masking import VideoTemporalAttentionMaskingHooker


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

def prepare_datasets_dataloaders(args):
    dataset, split = args.dataset, args.split
    
    # prepare metadata
    with open(f'./data/kinetics/{dataset}_{split}_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open(f'./data/kinetics/k400_val_metadata.pkl', 'rb') as f:
        k400_metadata = pickle.load(f)
    print('metadata prepared')
        
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
    
    # pts, fps = metadata['video_pts'], metadata['video_fps']
    # for pt in tqdm(pts):
    #     continue
    # k400_val_pts = k400_metadata['video_pts']
    # for pt in tqdm(k400_val_pts):
    #     continue
    
    start = time.time()
    print('loading kinetics...')
    kinetics_ds = VideoOnlyKinetics(
        root=dataset_root,
        frames_per_clip=16,
        split=split,
        num_workers=8,
        frame_rate=2,
        step_between_clips=1,
        transform=val_transform,
        _precomputed_metadata=metadata,  
    )
    print('elapsed:', time.time() - start)

    start = time.time()
    print('loading kinetics400 val...')
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
    print('elapsed:', time.time() - start)
    
    kinetics_dl = DataLoader(
        dataset=kinetics_ds,
        batch_size=32,
        shuffle=False,
        num_workers=12,
    )
    
    return kinetics_ds, k400_ds, kinetics_dl


def compute_masked_embeddings(args):
    dataset, split, force = args.dataset, args.split, args.force
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    mask_mode = args.mask_mode
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    print(f'computing {dataset} split {split}...')
    
    # embeddings_filename = f'./data/kinetics/{dataset}_{split}_embeddings.jsonl'
    embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'masked' / f'{dataset}_{split}'
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    assert spatial_masking or temporal_masking, 'at least one masking method should be selected'
    
    emb_fn = embeddings_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            emb_fn = emb_fn.parent / f'spatial_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.jsonl'
        else:
            emb_fn = emb_fn.parent / f'spatial_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.jsonl'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            emb_fn = emb_fn.parent / f'temporal_{emb_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.jsonl'
        else:
            emb_fn = emb_fn.parent / f'temporal_{emb_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.jsonl'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
    print('embeddings filename:', emb_fn)
         
    if not force and os.path.exists(emb_fn):
        return
    
    kinetics_ds, k400_ds, kinetics_dl = prepare_datasets_dataloaders(args)
    
    # prepare models
    model, cls_head = create_model()
    # print(model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cls_head = model.to(device), cls_head.to(device)
    
    # compute embeddings
    n_correct, n_total = 0, 0
    id, cache_rate = 0, 100
    logs = []
    
    # prepare precomputed rollout attention maps
    attn_maps_filename = f'./data/kinetics/attention_maps/{dataset}_{split}_{head_fusion}_{discard_ratio}_attn_maps.jsonl'
    spatial_attn_maps, temporal_attn_maps = [], []
    print('preparing precomputed rollout attention maps...')
    with open(attn_maps_filename, 'r') as f:
        for line in tqdm(f):
            attn_map_js = json.loads(line)
            spatial_attn_map = np.array(attn_map_js['spatial_attention_map'])
            spatial_attn_maps.append(spatial_attn_map)
            temporal_attn_map = np.array(attn_map_js['temporal_attention_map'])
            temporal_attn_maps.append(temporal_attn_map)
    
    # prepare config files
    config_path = Path('configs') / 'deit_tiny-pretrained-cifar10.yaml'
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
        config = config_dict.ConfigDict(config)
        
    # prepare attention extractor as a placeholder
    attention_extractor = FeatureExtractor(
        model=model,
        layer_name='attn_drop'
    )
    # do not need to hook
    # attention_extractor.hook()
    
    # prepare spatial and temporal attention masking
    if spatial_masking:
        spatial_attention_masking_hooker = VideoSpatialAttentionMaskingHooker(
            config=config,
            model=model,
            attention_extractor=attention_extractor,
            patch_embedding_layer_name='patch_embed.projection',
            mask_mode=mask_mode,
            mask_method=spatial_mask_method,
            mask_ratio=spatial_mask_ratio,
            mask_threshold=spatial_mask_threshold,
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )
        spatial_attention_masking_hooker.hook()
    
    if temporal_masking:
        temporal_attention_masking_hooker = VideoTemporalAttentionMaskingHooker(
            config=config,
            model=model,
            attention_extractor=attention_extractor,
            patch_embedding_layer_name='drop_before_time',
            mask_mode=mask_mode,
            mask_method=temporal_mask_method,
            mask_ratio=temporal_mask_ratio,
            mask_threshold=temporal_mask_threshold,
            head_fusion=head_fusion,
            discard_ratio=discard_ratio,
        )
        temporal_attention_masking_hooker.hook()
        

    model.eval()
    cls_head.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(kinetics_dl)):
            if batch_idx % cache_rate == 0:
                with open(emb_fn, 'a+') as f:
                    for log in logs:
                        f.write(json.dumps(log) + '\n')
                logs = []
                
            x, y = x.to(device), y.to(device)
            
            b = x.size(0)
            if spatial_masking:
                precomputed_attn_maps = spatial_attn_maps[b * batch_idx: b * (batch_idx + 1)]
                precomputed_attn_maps = np.concatenate(precomputed_attn_maps, axis=0)
            
                spatial_attention_masking_hooker.disable_masking()
                spatial_masks = spatial_attention_masking_hooker.generate_masks(
                    videos=x,
                    _precomputed_rollout_attention_maps=precomputed_attn_maps
                )
                # print('spatial mask:', spatial_masks.shape, np.where(spatial_masks == 1)[0].shape[0])
                spatial_attention_masking_hooker.enable_masking()
            
            if temporal_masking:
                precomputed_attn_maps = temporal_attn_maps[b * batch_idx: b * (batch_idx + 1)]
                precomputed_attn_maps = np.concatenate(precomputed_attn_maps, axis=0)
            
                temporal_attention_masking_hooker.disable_masking()
                temporal_masks = temporal_attention_masking_hooker.generate_masks(
                    videos=x,
                    _precomputed_rollout_attention_maps=precomputed_attn_maps
                )
                # print('temporal mask:', temporal_masks.shape, np.where(temporal_masks == 1)[0].shape[0])
                temporal_attention_masking_hooker.enable_masking()
                
            pre_logits = model(x)
            logits = cls_head(pre_logits)
            _, predicted = logits.max(1)

            for pre_logit, logit, gt, pred in zip(pre_logits, logits, y, predicted):
                pre_logit = pre_logit.detach().cpu().numpy().tolist()
                logit = logit.detach().cpu().numpy().tolist()
                gt_label = kinetics_ds.classes[gt.item()]
                pred_label = k400_ds.classes[pred.item()]
                logs.append({
                    'id': id, 
                    'gt': gt_label, 
                    'pred': pred_label, 
                    'penultimate': pre_logit,
                    'logit': logit,
                })
                id += 1

            n_total += y.size(0)
            n_correct += predicted.eq(y).sum().item()

        with open(emb_fn, 'a+') as f:
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
    
    parser.add_argument('--mask_mode', default='zero')
    
    parser.add_argument('--spatial_masking', action='store_true')
    parser.add_argument('--spatial_mask_method', type=str)
    parser.add_argument('--spatial_mask_ratio', type=float)
    parser.add_argument('--spatial_mask_threshold', type=float)
    
    parser.add_argument('--temporal_masking', action='store_true')
    parser.add_argument('--temporal_mask_method', type=str)
    parser.add_argument('--temporal_mask_ratio', type=float)
    parser.add_argument('--temporal_mask_threshold', type=float)
    
    parser.add_argument('--force', action='store_true')
    
    args = parser.parse_args()
    force = args.force
    compute_masked_embeddings(args)