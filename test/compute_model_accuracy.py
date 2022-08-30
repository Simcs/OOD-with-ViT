from pathlib import Path
import time
import pandas as pd
import argparse
import yaml
from pprint import pprint

from ml_collections import ConfigDict

from ood_with_vit.ood_classifier import Image_OOD_Classifier


datasets = [
    'CIFAR10',
    'CIFAR100',
]

models_to_configs = {
    # 'deit-tiny': {
    #     'CIFAR10': './configs/deit_tiny-pretrained-cifar10.yaml',
    #     'CIFAR100': './configs/deit_tiny-pretrained-cifar100.yaml',
    # },
    # 'deit-small': {
    #     'CIFAR10': './configs/deit_small-pretrained-cifar10.yaml',
    #     'CIFAR100': './configs/deit_small-pretrained-cifar100.yaml',
    # },
    # 'deit-base': {
    #     'CIFAR10': './configs/deit_base-pretrained-cifar10.yaml',
    #     'CIFAR100': './configs/deit_base-pretrained-cifar100.yaml',
    # },
    # 'vit-tiny': {
    #     'CIFAR10': './configs/vit_tiny-pretrained-cifar10.yaml',
    #     'CIFAR100': './configs/vit_tiny-pretrained-cifar100.yaml',
    # },
    # 'vit-small': {
    #     'CIFAR10': './configs/vit_small-pretrained-cifar10.yaml',
    #     'CIFAR100': './configs/vit_small-pretrained-cifar100.yaml',
    # },
    # 'vit-base': {
    #     'CIFAR10': './configs/vit_base-pretrained-cifar10.yaml',
    #     'CIFAR100': './configs/vit_base-pretrained-cifar100.yaml',
    # },
    'vit-tiny-in21k': {
        'CIFAR10': './configs/vit_tiny_in21k-pretrained-cifar10.yaml',
        'CIFAR100': './configs/vit_tiny_in21k-pretrained-cifar100.yaml',
    },
    'vit-small-in21k': {
        'CIFAR10': './configs/vit_small_in21k-pretrained-cifar10.yaml',
        'CIFAR100': './configs/vit_small_in21k-pretrained-cifar100.yaml',
    },
    'vit-base-in21k': {
        'CIFAR10': './configs/vit_base_in21k-pretrained-cifar10.yaml',
        'CIFAR100': './configs/vit_base_in21k-pretrained-cifar100.yaml',
    },
}


def compute_accruacies(args):
    acc_root = Path('./result') / 'accuracy'
    acc_root.mkdir(parents=True, exist_ok=True)
    for model, configs in models_to_configs.items():
        print(f'processing model {model}...')
        results = []
        for dataset_name, config_filename in configs.items():
            print(f'compute accuracy of {dataset_name}')
            # load yaml config and converts to ConfigDict
            with open(config_filename) as config_file:
                config = yaml.safe_load(config_file) 
                config = ConfigDict(config)
                ood_classifier = Image_OOD_Classifier(
                    config=config,
                    in_dist_dataset_name=dataset_name,
                    out_of_dist_dataset_name='SVHN',
                    log_dir=args.log_dir,
                )
                _, val_acc = ood_classifier.compute_accuracy()

                results.append([model, dataset_name, val_acc])
                print(f'result: model({model}), dataset({dataset_name}), accruacy({val_acc:.4f})')

            result_df = pd.DataFrame(
                data=results,
                columns=['model', 'dataset', 'accuracy'],
            )
            accuracy_filename = f'{model}_accuracy.tsv'
            accuracy_path = acc_root / accuracy_filename
            result_df.to_csv(accuracy_path, sep='\t')
            print(f'saved file {accuracy_path}!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Image OOD Detection')
    parser.add_argument('--finetuned', action='store_true', help='use finetuned model for evaluation')
    parser.add_argument('--log_dir', default='logs', type=str, help='training log directory')
    args = parser.parse_args()

    compute_accruacies(args)
