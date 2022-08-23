from pathlib import Path
import time
import pandas as pd
import argparse
import yaml
from pprint import pprint

from ml_collections import ConfigDict

from ood_with_vit.ood_classifier import Image_OOD_Classifier


datasets = ['CIFAR10', 'CIFAR100', 'SVHN']
pretrain_only_metrics = ['Mahalanobis', 'ClasswiseMahalanobis', 'DMD', 'DCMD']
baseline_metrics = ['MSP', 'Mahalanobis', 'ClasswiseMahalanobis', 'SML']
mask_metrics = ['DML', 'DCL', 'DMD', 'DCMD']
mask_metrics = ['DMD']
mask_metrics = ['DCMD']
models_to_configs = {
    'deit-tiny': {
        'CIFAR10': './configs/deit_tiny-pretrained-cifar10.yaml',
        'CIFAR100': './configs/deit_tiny-pretrained-cifar100.yaml',
    }
}

ratio_methods = ['top_ratio', 'bottom_ratio', 'random']
threshold_methods = ['gt_threshold', 'lt_threshold']
mask_methods = ratio_methods + threshold_methods
n_params = 20
mask_ratios = [1 / n_params * i for i in range(n_params + 1)]
mask_thresholds = [1 / n_params * i  for i in range(n_params + 1)]

masking_params = []
for method in ratio_methods:
    for ratio in mask_ratios:
        masking_params.append({
            'mask_method': method,
            'mask_ratio': ratio,
            'mask_threshold': None,
        })

for method in threshold_methods:
    for threshold in mask_thresholds:
        masking_params.append({
            'mask_method': method,
            'mask_ratio': None,
            'mask_threshold': threshold,
        })


def test_one(config, args):
    ood_classifier = Image_OOD_Classifier(
        config=config,
        in_dist_dataset_name=args.id_dataset_name,
        out_of_dist_dataset_name=args.ood_dataset_name,
        log_dir=args.log_dir
    )
    result = ood_classifier.compute_ood_classification_results()
    pprint(result)


def test_baseline_metrics(args):
    ood_root = Path('./result') / 'ood_scores'
    ood_root.mkdir(parents=True, exist_ok=True)
    for model, configs in models_to_configs.items():
        print(f'processing model {model}...')
        for id_dataset_name, config_filename in configs.items():
            for ood_dataset_name in datasets:
                if id_dataset_name == ood_dataset_name:
                    continue

                print(f'ID: {id_dataset_name}, OOD: {ood_dataset_name}')
                # load yaml config and converts to ConfigDict
                with open(config_filename) as config_file:
                    config = yaml.safe_load(config_file) 
                    config = ConfigDict(config)

                results = []
                for metric_name in baseline_metrics:
                    if not args.finetuned and metric_name not in pretrain_only_metrics:
                        continue

                    ood_classifier = Image_OOD_Classifier(
                        config=config,
                        in_dist_dataset_name=id_dataset_name,
                        out_of_dist_dataset_name=ood_dataset_name,
                        log_dir=args.log_dir,       
                    )
                    auroc_score, aupr_score, fpr95 = ood_classifier.compute_ood_classification_results(
                        metric_name=metric_name,
                        finetuned=args.finetuned,
                    )
                    result = [metric_name, auroc_score, aupr_score, fpr95]
                    results.append(result)
                    print(f'result: auroc({auroc_score:.4f}), aupr({aupr_score:.4f}), fpr95({fpr95:.4f})\n')

                result_df = pd.DataFrame(
                    data=results,
                    columns=['metric', 'auroc_score', 'aupr_score', 'fpr95'],
                )
                model_name = ood_classifier.model_name
                id_name = ood_classifier.in_dist_dataset_name
                ood_name = ood_classifier.out_of_dist_dataset_name
                finetuned = 'finetuned' if args.finetuned else 'pretrain-only'
                ood_score_filename = f'{model_name}_{id_name}_vs_{ood_name}_{finetuned}_baseline.csv'
                ood_score_path = ood_root / ood_score_filename
                result_df.to_csv(ood_score_path, sep=',')


def test_mask_metrics(args):
    ood_root = Path('./result') / 'ood_scores'
    ood_root.mkdir(parents=True, exist_ok=True)
    for model, configs in models_to_configs.items():
        print(f'processing model {model}...')
        for id_dataset_name, config_filename in configs.items():
            for ood_dataset_name in datasets:
                if id_dataset_name == ood_dataset_name:
                    continue

                print(f'ID: {id_dataset_name}, OOD: {ood_dataset_name}')
                # load yaml config and converts to ConfigDict
                with open(config_filename) as config_file:
                    config = yaml.safe_load(config_file) 
                    config = ConfigDict(config)

                for metric_name in mask_metrics:
                    results = []
                    if not args.finetuned and metric_name not in pretrain_only_metrics:
                        continue

                    ood_classifier = Image_OOD_Classifier(
                        config=config,
                        in_dist_dataset_name=id_dataset_name,
                        out_of_dist_dataset_name=ood_dataset_name,
                        log_dir=args.log_dir,       
                    )
                    for masking_param in masking_params:
                        print('params:', masking_param)
                        mask_method = masking_param['mask_method']
                        mask_ratio = masking_param['mask_ratio']
                        mask_threshold = masking_param['mask_threshold']
                        auroc_score, aupr_score, fpr95 = ood_classifier.compute_ood_classification_results(
                            metric_name=metric_name,
                            finetuned=args.finetuned,
                            mask_method=mask_method,
                            mask_ratio=mask_ratio,
                            mask_threshold=mask_threshold,
                        )
                        result = [metric_name, mask_method, mask_ratio, mask_threshold, \
                            auroc_score, aupr_score, fpr95]
                        results.append(result)
                        print(f'result: auroc({auroc_score:.4f}), aupr({aupr_score:.4f}), fpr95({fpr95:.3f})\n')

                    result_df = pd.DataFrame(
                        data=results,
                        columns=['metric', 'mask_method', 'mask_ratio', 'mask_threshold', 'auroc_score', 'aupr_score', 'fpr95'],
                    )
                    model_name = ood_classifier.model_name
                    id_name = ood_classifier.in_dist_dataset_name
                    ood_name = ood_classifier.out_of_dist_dataset_name
                    finetuned = 'finetuned' if args.finetuned else 'pretrain-only'
                    ood_score_filename = f'{model_name}_{id_name}_vs_{ood_name}_{finetuned}_{metric_name}_mask.csv'
                    ood_score_path = ood_root / ood_score_filename
                    result_df.to_csv(ood_score_path, sep=',')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Image OOD Detection')
    parser.add_argument('--config', type=str, help='config filename')
    parser.add_argument('--id_dataset_name', type=str, help='in-distribution dataset name')
    parser.add_argument('--ood_dataset_name', type=str, help='out-of-distribution dataset name')
    parser.add_argument('--test_mode', choices=['baseline', 'mask', 'all'], help='select test metrics')
    parser.add_argument('--finetuned', action='store_true', help='use finetuned model for evaluation')
    parser.add_argument('--log_dir', default='logs', type=str, help='training log directory')
    args = parser.parse_args()

    if args.test_mode == 'baseline':
        test_baseline_metrics(args)
    elif args.test_mode == 'mask':
        test_mask_metrics(args)
    elif args.test_mode == 'all':
        test_baseline_metrics(args)
        test_mask_metrics(args)
    else:
        assert args.config and args.id_dataset_name and args.ood_dataset_name
        # load yaml config and converts to ConfigDict
        with open(args.config) as config_file:
            config = yaml.safe_load(config_file) 
            config = ConfigDict(config)
        test_one(config, args)

