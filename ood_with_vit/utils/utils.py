from re import L
import torch

from ml_collections import ConfigDict

from ood_with_vit.visualizer.feature_extractor import FeatureExtractor


def compute_attention_maps(
    config: ConfigDict,
    model: torch.nn.Module,
    imgs: torch.Tensor,
):
    # compute outputs and penultimate features detached and moved to cpu.
    if config.model.pretrained:
        # use forward hook to get attnetion maps of each layer
        feature_extractor = FeatureExtractor(
            model=model,
            layer_name=config.model.layer_name.attention,
        )
        _ = feature_extractor(imgs)
        attention_maps = []
        for attention_map in feature_extractor.features:
            attention_maps.append(attention_map)
    else:
        _, _attention_maps = model(imgs)
        attention_maps = []
        for attention_map in _attention_maps:
            attention_maps.append(attention_map)
    
    return attention_maps

def compute_penultimate_features(
    config: ConfigDict,
    model: torch.nn.Module,
    imgs: torch.Tensor,
):
    # compute penultimate features detached and moved to cpu.
    if config.model.pretrained:
        feature_extractor = FeatureExtractor(
            model=model,
            layer_name=config.model.layer_name.penultimate,
        )
        outputs = feature_extractor(imgs)
        penultimate_features = feature_extractor.features[0]
    else:
        outputs, penultimate_features = model.get_penultimate_features(imgs)
        # penultimate_features = penultimate_features
    
    return penultimate_features


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