import json
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import random

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve

random.seed(1234)

def get_graph_filename(
    dataset1: str,
    dataset2: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    image_dir = Path('./result') / 'images' / 'masked' / f'{dataset1}_vs_{dataset2}'
    image_dir.mkdir(exist_ok=True)
    image_fn = image_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            image_fn = image_fn.parent / f'spatial_{image_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}'
        else:
            image_fn = image_fn.parent / f'spatial_{image_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            image_fn = image_fn.parent / f'temporal_{image_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}'
        else:
            image_fn = image_fn.parent / f'temporal_{image_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}')
            else:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}')
        else:
            if 'ratio' in temporal_mask_method:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}')
            else:
                image_fn = image_fn.parent / (f'spatiotemporal_{image_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}')
                
    return image_fn

def get_result_filename(
    dataset1: str,
    dataset2: str,
    args,
):
    head_fusion, discard_ratio = args.head_fusion, args.discard_ratio
    spatial_masking, temporal_masking = args.spatial_masking, args.temporal_masking
    spatial_mask_method = args.spatial_mask_method
    spatial_mask_ratio, spatial_mask_threshold = args.spatial_mask_ratio, args.spatial_mask_threshold
    temporal_mask_method = args.temporal_mask_method
    temporal_mask_ratio, temporal_mask_threshold = args.temporal_mask_ratio, args.temporal_mask_threshold
    
    result_dir = Path('./result') / 'ood_scores' / 'video' / 'masked' / f'{dataset1}_vs_{dataset2}'
    result_dir.mkdir(exist_ok=True)
    result_fn = result_dir / f'{head_fusion}_{discard_ratio}'
    if spatial_masking and not temporal_masking:
        if 'ratio' in spatial_mask_method:
            result_fn = result_fn.parent / f'spatial_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.jsonl'
        else:
            result_fn = result_fn.parent / f'spatial_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.jsonl'
    elif not spatial_masking and temporal_masking:
        if 'ratio' in temporal_mask_method:
            result_fn = result_fn.parent / f'temporal_{result_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.jsonl'
        else:
            result_fn = result_fn.parent / f'temporal_{result_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.jsonl'
    elif spatial_masking and temporal_masking:
        if 'ratio' in spatial_mask_method:
            if 'ratio' in temporal_mask_method:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                    f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return result_fn

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
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                emb_fn = emb_fn.parent / (f'spatiotemporal_{emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
    return emb_fn

def compute_msp_ood_scrores(args):
    # kinetics400 vs. kinetics600
    # compute k400 mahalanobis distances
    print('loading k400 embeddings...')
    k400_embeddings_filename = get_embeddings_filename('k400', 'val', args)
    k400_val = []
    with open(k400_embeddings_filename, 'r') as f:
        for line in tqdm(f):
            emb_js = json.loads(line)
            logit = torch.Tensor(emb_js['logit'])
            k400_val.append(logit.view(1, -1))
    k400_val = torch.cat(k400_val, dim=0)
    k400_msp, _ = k400_val.max(dim=1)
    k400_msp = (-k400_msp.numpy()).tolist()
        
    # compute k600 mahalanobis distances
    print('loading k600 embeddings...')
    k600_emb_fn = get_embeddings_filename('k600', 'val', args)
    k600_embeddings_filename = k600_emb_fn.parent / (k600_emb_fn.stem + '_deduplicated' + k600_emb_fn.suffix)
    k600_original = []
    with open(k600_embeddings_filename, 'r') as f:
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
    result_filename = get_result_filename('k400', 'k600', args)
    result_filename = result_filename.parent / (result_filename.stem + '_msp' + result_filename.suffix)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': k600_auroc_score,
            'aupr': k600_aupr_score,
            'fpr95': k600_fpr95,
        }))
    
    print('auroc:', k600_auroc_score, 'aupr:', k600_aupr_score, 'fpr95:', k600_fpr95)

    image_filename = get_graph_filename('k400', 'k600', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_msp_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_msp_aupr.png')
    save_precision_recall_curve(pr, re, aupr_filename)

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
    result_filename = get_result_filename('k400', 'k700', args)
    result_filename = result_filename.parent / (result_filename.stem + '_msp' + result_filename.suffix)
    
    print('k400 vs. k700:')
    print('auroc:', k700_auroc_score, 'aupr:', k700_aupr_score, 'fpr95:', k700_fpr95)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': k700_auroc_score,
            'aupr': k700_aupr_score,
            'fpr95': k700_fpr95,
        }))
    image_filename = get_graph_filename('k400', 'k700', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_msp_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_msp_aupr.png')
    save_precision_recall_curve(pr, re, aupr_filename)
    
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
    
    compute_msp_ood_scrores(args)