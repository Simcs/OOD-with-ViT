import time
import argparse
import yaml
from pprint import pprint

from ml_collections import ConfigDict

from ood_with_vit.ood_classifier import Image_OOD_Classifier

parser = argparse.ArgumentParser(description='PyTorch Image OOD Detection')
parser.add_argument('--config', type=str, help='config filename')
parser.add_argument('--id_dataset_name', type=str, help='in-distribution dataset name')
parser.add_argument('--ood_dataset_name', type=str, help='out-of-distribution dataset name')
parser.add_argument('--test_all', action='store_true', help='test every possible combinations of id/ood dataset')
parser.add_argument('--log_dir', default='logs', type=str, help='training log directory')
args = parser.parse_args()

datasets = ['CIFAR10', 'CIFAR100', 'SVHN']
models_to_configs = {
    'deit-tiny': {
        'CIFAR10': './configs/deit_tiny-pretrained-cifar10.yaml',
        'CIFAR100': './configs/deit_tiny-pretrained-cifar100.yaml',
    }
}

def test_one(config, args):
    ood_classifier = Image_OOD_Classifier(
        config=config,
        in_dist_dataset_name=args.id_dataset_name,
        out_of_dist_dataset_name=args.ood_dataset_name,
        log_dir=args.log_dir
    )
    result = ood_classifier.compute_ood_classification_results()
    pprint(result)

def test_all(args):
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

                ood_classifier = Image_OOD_Classifier(
                    config=config,
                    in_dist_dataset_name=id_dataset_name,
                    out_of_dist_dataset_name=ood_dataset_name,
                    log_dir=args.log_dir                    
                )
                result = ood_classifier.compute_ood_classification_results()
                pprint(result)

if __name__ == "__main__":

    if args.test_all:
        test_all(args)
    else:
        assert args.config and args.id_dataset_name and args.ood_dataset_name
        # load yaml config and converts to ConfigDict
        with open(args.config) as config_file:
            config = yaml.safe_load(config_file) 
            config = ConfigDict(config)
        test_one(config, args)

