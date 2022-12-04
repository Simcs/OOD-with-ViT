import json
import pickle
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import random

from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.utils.visualization import save_roc_curve, save_precision_recall_curve


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
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        else:
            if 'ratio' in temporal_mask_method:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
            else:
                result_fn = result_fn.parent / (f'spatiotemporal_{result_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_'\
                    + f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                
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


def compute_mahalanobis_ood_scores(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading train statistics...')
    stat_fn = get_statistics_filename('k400', 'train', args)
    stat_fn = stat_fn.parent / f'{stat_fn.stem}_statistics_total.pkl'
    stat_fn = 'data/kinetics/statistics/masked/k400_train/spatial_max_0.5_lt_threshold_0.01_statistics_total.pkl'
    print('train stat:', stat_fn)
    with open(stat_fn, 'rb') as f:
        train_stat_total = pickle.load(f)

    # prepare statistics
    means, precision = train_stat_total['mean'], train_stat_total['total_precision']
    for k400_class in means:
        means[k400_class] = torch.Tensor(means[k400_class]).to(device)
    precision = torch.Tensor(precision).to(device).float()

    # kinetics400 vs. kinetics600
    # compute k400 mahalanobis distances
    print('loading k400 embeddings...')
    k400_embeddings_filename = get_embeddings_filename('k400', 'val', args)
    k400_val = []
    with open(k400_embeddings_filename, 'r') as f:
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
    k600_emb_fn = get_embeddings_filename('k600', 'val', args)
    k600_embeddings_filename = k600_emb_fn.parent / (k600_emb_fn.stem + '_deduplicated' + k600_emb_fn.suffix)
    k600_original = []
    with open(k600_embeddings_filename, 'r') as f:
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
    # max_ood_score = max(ood_scores)
    # ood_scores = [i / max_ood_score for i in ood_scores]
    print('# of k400 and k600 embeddings:', len(set(ood_scores)))

    measure_performances(k400_mahalanobis_distances, k600_mahalanobis_distances)

    fpr, tpr, k600_auroc_score = auroc(test_y, ood_scores)
    pr, re, k600_aupr_score = aupr(test_y, ood_scores)
    k600_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
    result_filename = get_result_filename('k400', 'k600', args)
    result_filename = result_filename.parent / (result_filename.stem + '_maha' + result_filename.suffix)
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': k600_auroc_score,
            'aupr': k600_aupr_score,
            'fpr95': k600_fpr95,
        }))
    
    print('k400 vs. k600:')
    print('auroc:', k600_auroc_score, 'aupr:', k600_aupr_score, 'fpr95:', k600_fpr95)
    image_filename = get_graph_filename('k400', 'k600', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_maha_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_maha_aupr.png')
    save_precision_recall_curve(pr, re, aupr_filename)


    # compute k700 mahalanobis distances
    print('loading k700 embeddings...')
    k700_emb_fn = get_embeddings_filename('k700-2020', 'val', args)
    k700_embeddings_filename = k700_emb_fn.parent / (k600_emb_fn.stem + '_deduplicated' + k700_emb_fn.suffix)
    k700_original = []
    with open(k700_embeddings_filename, 'r') as f:
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
    # max_ood_score = max(ood_scores)
    # ood_scores = [i / max_ood_score for i in ood_scores]
    print('# of k400 and k700 embeddings:', len(set(ood_scores)))

    fpr, tpr, k700_auroc_score = auroc(test_y, ood_scores)
    pr, re, k700_aupr_score = aupr(test_y, ood_scores)
    k700_fpr95 = fpr_at_95_tpr(test_y, ood_scores)
    result_filename = get_result_filename('k400', 'k700', args)
    result_filename = result_filename.parent / (result_filename.stem + '_maha' + result_filename.suffix)
    
    with open(result_filename, 'w') as f:
        f.write(json.dumps({
            'auroc': k700_auroc_score,
            'aupr': k700_aupr_score,
            'fpr95': k700_fpr95,
        }))
        
    print('k400 vs. k700:')
    print('auroc:', k700_auroc_score, 'aupr:', k700_aupr_score, 'fpr95:', k700_fpr95)
    image_filename = get_graph_filename('k400', 'k700', args)
    auroc_filename = image_filename.parent / (image_filename.name + '_maha_auroc.png')
    save_roc_curve(fpr, tpr, auroc_filename)
    aupr_filename = image_filename.parent / (image_filename.name + '_maha_aupr.png')
    save_precision_recall_curve(pr, re, aupr_filename)

    measure_performances(k400_mahalanobis_distances, k700_mahalanobis_distances)


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
    
    compute_mahalanobis_ood_scores(args)