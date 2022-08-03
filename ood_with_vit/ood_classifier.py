from typing import Tuple
from collections import OrderedDict
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from ml_collections import ConfigDict

from ood_with_vit.models.vit import ViT
from ood_with_vit.metrics import MSP, Mahalanobis, SML
from ood_with_vit.utils import compute_ood_scores
from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr


DATASETS = ['CIFAR10', 'CIFAR100', 'SVHN']

class Image_OOD_Classifier:

    def __init__(
        self,
        config: ConfigDict,
        in_dist_dataset_name: str,
        out_of_dist_dataset_name: str,
        log_dir: str):

        assert in_dist_dataset_name != out_of_dist_dataset_name, 'ID and OOD dataset must be different.'
        assert in_dist_dataset_name in DATASETS, f'dataset {in_dist_dataset_name} is not supported.'
        assert out_of_dist_dataset_name in DATASETS, f'dataset {out_of_dist_dataset_name} is not supported.'

        self.config = config
        self.in_dist_dataset_name = in_dist_dataset_name
        self.out_of_dist_dataset_name = out_of_dist_dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # frequently used variables
        self.model_name = config.model.name
        self.summary = config.summary

        self.log_root = Path(log_dir) / self.model_name / self.summary
        self.ood_root = Path('./result') / 'ood_scores'
        self.ood_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.log_root / 'checkpoints'

        # create transform for images
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        img_size = self.config.model.img_size

        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])

        # create model and test dataloaders
        self.model = self._create_model()
        self.id_test_dataloader, self.ood_test_dataloader = self._create_dataloaders()
    
    def _create_model(self) -> torch.nn.Module:
        summary = self.config.summary
        n_class = self.config.dataset.n_class

        if self.config.model.pretrained:
            model = torch.hub.load(
                repo_or_dir=self.config.model.repo,
                model=self.config.model.pretrained_model,
                pretrained=False,
            )
            model.head = nn.Linear(model.head.in_features, n_class)
        else:
            model = ViT(
                image_size=self.config.model.img_size,
                patch_size=self.config.model.patch_size,
                num_classes=n_class,
                dim=self.config.model.dim_head,
                depth=self.config.model.depth,
                heads=self.config.model.n_heads,
                mlp_dim=self.config.model.dim_mlp,
                dropout=self.config.model.dropout,
                emb_dropout=self.config.model.emb_dropout,
                visualize=True,
            )

        model = model.to(device=self.device)

        checkpoint = torch.load(self.checkpoint_path / f'{summary}_best.pt')

        state_dict = checkpoint['model_state_dict']
        trimmed_keys = []
        for key in state_dict.keys():
            # remove prefix 'module.' for each key (in case of DataParallel)
            trimmed_keys.append(key[7:])
        trimmed_state_dict = OrderedDict(list(zip(trimmed_keys, state_dict.values())))

        model.load_state_dict(trimmed_state_dict)

        return model
    
    def _create_dataset(self, dataset_name: str, train: bool) -> Dataset:
        dataset_root = self.config.dataset.root
        if dataset_name == 'CIFAR10':
            dataset = CIFAR10(
                root=dataset_root, 
                train=train, 
                download=False, 
                transform=self.transform_test
            )
        elif dataset_name == 'CIFAR100':
            dataset = CIFAR100(
                root=dataset_root, 
                train=train, 
                download=False, 
                transform=self.transform_test
            )
        else:
            split = 'train' if train else 'test'
            dataset = SVHN(
                root=dataset_root, 
                split=split, 
                download=True, 
                transform=self.transform_test
            )
        return dataset

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        id_test_dataset = self._create_dataset(self.in_dist_dataset_name, False)
        id_test_dataloader = DataLoader(
            dataset=id_test_dataset,
            batch_size=self.config.eval.batch_size,
            shuffle=False,
            num_workers=8,
        )
        ood_test_dataset = self._create_dataset(self.out_of_dist_dataset_name, False)
        ood_test_dataloader = DataLoader(
            dataset=ood_test_dataset,
            batch_size=self.config.eval.batch_size,
            shuffle=False,
            num_workers=8,
        )
        return id_test_dataloader, ood_test_dataloader

    def compute_ood_classification_results(self):
        id_train_dataset = self._create_dataset(self.in_dist_dataset_name, True)
        id_train_dataloader = DataLoader(
            dataset=id_train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=8,
        )

        metrics = {
            'MSP': MSP(self.config, self.model),
            'Mahalanobis': Mahalanobis(self.config, self.model, id_train_dataloader),
            'SML': SML(self.config, self.model, id_train_dataloader),
        }


        scores = {}
        score_list = []
        for name, metric in metrics.items():
            print(f'Compute ood scores by {name}')
            test_y, ood_scores, id_ood_scores, ood_ood_scores = compute_ood_scores(
                metric=metric,
                in_dist_dataloader=self.id_test_dataloader,
                out_of_dist_dataloader=self.ood_test_dataloader,
            )
            _, _, auroc_score = auroc(test_y, ood_scores)
            _, _, aupr_score = aupr(test_y, ood_scores)
            fpr95 = fpr_at_95_tpr(test_y, ood_scores)
            scores[name] = {
                'auroc': auroc_score,
                'aupr': aupr_score,
                'fpr_at_95_tpr': fpr95,
            }
            score_list.append([auroc_score, aupr_score, fpr95])

        result_df = pd.DataFrame(
            data=score_list,
            index=metrics.keys(),
            columns=['auroc', 'aupr', 'fpr95'],
        )
        results = {
            'in_dist_dataset': self.in_dist_dataset_name,
            'out_of_dist_dataset': self.out_of_dist_dataset_name,
            'scores': scores
        }

        ood_score_filename = f'{self.model_name}_{self.in_dist_dataset_name}_vs_{self.out_of_dist_dataset_name}.csv'
        ood_score_path = self.ood_root / ood_score_filename
        result_df.to_csv(ood_score_path, sep=',')
        return results