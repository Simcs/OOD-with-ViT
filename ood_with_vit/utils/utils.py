from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from ml_collections import ConfigDict

from ood_with_vit.visualizer.feature_extractor import FeatureExtractor

if TYPE_CHECKING:
    from ood_with_vit.metrics import Metric


def compute_attention_maps(
    config: ConfigDict,
    model: torch.nn.Module,
    input: torch.Tensor,
    feature_extractor: Optional[FeatureExtractor] = None,
):
    # compute outputs and penultimate features detached and moved to cpu.
    if config.model.pretrained:
        assert feature_extractor is not None, 'feature_extractor must exist'
        _ = feature_extractor(input)
        attention_maps = []
        for attention_map in feature_extractor.features:
            attention_maps.append(attention_map)
    else:
        _, _attention_maps = model(input)
        attention_maps = []
        for attention_map in _attention_maps:
            attention_maps.append(attention_map)
    
    return attention_maps


def compute_penultimate_features(
    config: ConfigDict,
    model: torch.nn.Module,
    imgs: torch.Tensor,
    feature_extractor: Optional[FeatureExtractor] = None,
):
    # compute penultimate features detached and moved to cpu.
    if config.model.pretrained:
        assert feature_extractor is not None, 'feature_extractor must exist'
        _ = feature_extractor(imgs)
        penultimate_features = feature_extractor.features[0]
    else:
        _, penultimate_features = model.get_penultimate_features(imgs)
    
    return penultimate_features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_logits(
    config: ConfigDict,
    model: torch.nn.Module,
    imgs: torch.Tensor,
):
    # compute logits detached and moved to cpu.
    if config.model.pretrained:
        logits = model(imgs)
    else:
        logits, _ = model(imgs)
    logits = logits.detach().cpu()
    return logits


def compute_ood_scores(
    metric: Metric,
    in_dist_dataloader: DataLoader,
    out_of_dist_dataloader: DataLoader,
):
    # seed = 3456
    # set_seed(seed)
    # print(f'set seed {seed}')
    test_y, ood_scores = [], []
    print('processing in-distribution samples...')
    id_ood_scores = metric.compute_dataset_ood_score(in_dist_dataloader)
    print('processing out-of-distribution samples...')   
    ood_ood_scores = metric.compute_dataset_ood_score(out_of_dist_dataloader)

    min_len = min(len(id_ood_scores), len(ood_ood_scores))
    id_ood_scores = random.sample(id_ood_scores, min_len)
    ood_ood_scores = random.sample(ood_ood_scores, min_len)
    test_y = [0 for _ in range(len(id_ood_scores))] + [1 for _ in range(len(ood_ood_scores))]
    ood_scores = id_ood_scores + ood_ood_scores

    return test_y, ood_scores, id_ood_scores, ood_ood_scores

