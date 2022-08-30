import argparse
import yaml
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from ml_collections import ConfigDict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from ood_with_vit.models.vit import ViT
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor
from ood_with_vit.mim.attention_masking import AttentionMaskingHooker


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_mask_on_img(img, mask):
    img_h, img_w, _ = img.shape
    mask = cv2.resize(mask, (img_h, img_w), interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, axis=-1)
    masked_img = img * (1 - mask)
    return masked_img


def initialize_vit_model(config, finetuned: bool = True, verbose=0):
    assert config.model.pretrained, 'only pretrained models are allowed'

    # frequently used variables
    model_name = config.model.name
    patch_size = config.model.patch_size
    summary = config.summary

    # log directories
    log_root = Path('./logs') / model_name / summary
    checkpoint_path = log_root / 'checkpoints'

    if config.model.pretrained:
        if finetuned:
            print('init finetuned model')
            n_class = config.dataset.n_class
            model = torch.hub.load(
                repo_or_dir=config.model.repo,
                model=config.model.pretrained_model,
                pretrained=False,
            )
            model.head = nn.Linear(model.head.in_features, n_class)
        else:
            print('init pretrained-only model')
            model = torch.hub.load(
                repo_or_dir=config.model.repo,
                model=config.model.pretrained_model,
                pretrained=True,
            )
    model = model.to(device=device)
    if verbose:
        print(model)
        
    if finetuned:
        checkpoint = torch.load(checkpoint_path / f'{summary}_best.pt')

        state_dict = checkpoint['model_state_dict']
        trimmed_keys = []
        for key in state_dict.keys():
            # remove prefix 'module.' for each key (in case of DataParallel)
            trimmed_keys.append(key[7:])
        trimmed_state_dict = OrderedDict(list(zip(trimmed_keys, state_dict.values())))

        model.load_state_dict(trimmed_state_dict)
    return model


def create_dataset(config, dataset_name: str, train: bool):
    dataset_mean, dataset_std = config.dataset.mean, config.dataset.std
    dataset_root = config.dataset.root
    img_size = config.model.img_size

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
    ])      
    if dataset_name == 'cifar10':
        dataset = CIFAR10(
            root=dataset_root, 
            train=train, 
            download=False, 
            transform=transform_test,
        )
    elif dataset_name == 'cifar100':
        dataset = CIFAR100(
            root=dataset_root, 
            train=train, 
            download=False, 
            transform=transform_test,
        )
    return dataset


def create_hookers(config):
    model = initialize_vit_model(config, finetuned=False, verbose=1)

    # add hooks for feature extraction
    feature_extractor = FeatureExtractor(
        model=model,
        layer_name=config.model.layer_name.penultimate,
    )
    feature_extractor.hook()
    
    # add hooks for attention map extraction
    attention_extractor = FeatureExtractor(
        model=model,
        layer_name=config.model.layer_name.attention,
    )
    attention_extractor.hook()

    # add hooks for attention masking
    mask_method = 'lt_threshold'
    mask_ratio = 0.3
    mask_threshold = 0.1
    head_fusion = 'max'
    discard_ratio = 0.9

    attention_masking_hooker = AttentionMaskingHooker(
        config=config,
        model=model,
        attention_extractor=attention_extractor,
        patch_embedding_layer_name='patch_embed.norm',
        mask_method=mask_method,
        mask_ratio=mask_ratio,
        mask_threshold=mask_threshold,
        head_fusion=head_fusion,
        discard_ratio=discard_ratio,
    )
    attention_masking_hooker.hook()

    return feature_extractor, attention_extractor, attention_masking_hooker


def generate_masked_dataset(config, dataset_name):
    feature_extractor, attention_extractor, attention_masking_hooker = create_hookers(config)

    dataset = create_dataset(config, dataset_name, True)
    print(dataset.classes)
    dataset_path = Path('./data') / f'masked_{dataset_name}' / 'train'
    dataset_path.mkdir(parents=True, exist_ok=True)
    for cls in dataset.classes:
        cls_path = dataset_path / cls
        cls_path.mkdir(parents=True, exist_ok=True)

    for cls_idx in range(len(dataset.classes)):
        img_indicies = np.where(np.array(dataset.targets) == cls_idx)[0]
        for i in tqdm(img_indicies):
            img, _ = dataset[i]
            img = img.to(device)

            attention_masking_hooker.disable_masking()
            masks = attention_masking_hooker.generate_masks(img.unsqueeze(0))
            
            original_img_path = f'./assets/{dataset_name}/train/{i:05d}.jpg'
            original_img = cv2.imread(original_img_path)
            masked_img_path = dataset_path / dataset.classes[cls_idx] / f'{i:05d}.jpg'
            masked_img = apply_mask_on_img(original_img, masks[0])

            cv2.imwrite(str(masked_img_path), masked_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Image OOD Detection')
    parser.add_argument('--config', type=str, help='config filename')
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    parser.add_argument('--test_mode', choices=['baseline', 'mask', 'all'], help='select test metrics')
    parser.add_argument('--finetuned', action='store_true', help='use finetuned model for evaluation')
    parser.add_argument('--log_dir', default='logs', type=str, help='training log directory')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file) 
        config = ConfigDict(config)

    generate_masked_dataset(config, args.dataset_name)
