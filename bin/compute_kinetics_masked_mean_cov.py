import json
import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

from torchvision.datasets import Kinetics
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance

def get_embeddings_filename(
    dataset: str,
    split: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'masked' / f'{dataset}_{split}'
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
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return emb_fn

def get_statistics_filename(
    dataset: str,
    split: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    statistics_dir = Path('./data') / 'kinetics' / 'statistics' / 'masked' / f'{dataset}_{split}'
    stat_fn = statistics_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            stat_fn = stat_fn.parent / f'spatial_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.pkl'
        else:
            stat_fn = stat_fn.parent / f'spatial_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.pkl'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            stat_fn = stat_fn.parent / f'temporal_{stat_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.pkl'
        else:
            stat_fn = stat_fn.parent / f'temporal_{stat_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.pkl'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.pkl')
            else:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.pkl')
        else:
            if 'ratio' in temporal_mask_method:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.pkl')
            else:
                stat_fn = stat_fn.parent / (f'spatiotemporal_{stat_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.pkl')
                
    return stat_fn

def compute_masked_statistics(args):
    correct_penultimates = {}
    # wrong_penultimates = {}

    embeddings_filename = get_embeddings_filename('k400', 'train', args)
    print('embeddings filename:', embeddings_filename)
    # filename = Path('./data/kinetics/embeddings/masked/k400_train/spatial_max_0.5_lt_threshold_0.01.jsonl')
    # print(filename.parent)
    # print(filename.stem)
    statistics_filename = get_statistics_filename('k400', 'train', args)
    total_filename = statistics_filename.parent / f'{statistics_filename.stem}_statistics_total.pkl'
    classwise_filename = statistics_filename.parent / f'{statistics_filename.stem}_statistics_classwise.pkl'
    print('statistics:', statistics_filename)
    print('total:', total_filename)
    print('classwise:', classwise_filename)
    
    with open(embeddings_filename, 'r') as f:
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

    statistics_filename = get_statistics_filename('k400', 'train', args)
    total_filename = statistics_filename.parent / f'{statistics_filename.stem}_statistics_total.pkl'
    with open(total_filename, 'wb') as f:
        pickle.dump(result_total, f)

    classwise_filename = statistics_filename.parent / f'{statistics_filename.stem}_statistics_classwise.pkl'
    with open(classwise_filename, 'wb') as f:
        pickle.dump(result_classwise, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--head_fusion', default='max')
    parser.add_argument('--discard_ratio', default=0.5)
    
    parser.add_argument('--spatial_masking', action='store_true')
    parser.add_argument('--spatial_mask_method', type=str)
    parser.add_argument('--spatial_mask_ratio', type=float)
    parser.add_argument('--spatial_mask_threshold', type=float)
    
    parser.add_argument('--temporal_masking', action='store_true')
    parser.add_argument('--temporal_mask_method', type=str)
    parser.add_argument('--temporal_mask_ratio', type=float)
    parser.add_argument('--temporal_mask_threshold', type=float)
    
    args = parser.parse_args()
    
    compute_masked_statistics(args)