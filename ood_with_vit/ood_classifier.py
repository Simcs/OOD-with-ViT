from typing import Tuple
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from ml_collections import ConfigDict

from ood_with_vit.models.vit import ViT
from ood_with_vit.metrics import MSP, Mahalanobis, ClasswiseMahalanobis, SML
from ood_with_vit.metrics import DCL, DML, DMD, DCMD
from ood_with_vit.utils import compute_ood_scores
from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor


DATASETS = ['CIFAR10', 'CIFAR100', 'SVHN']

class Image_OOD_Classifier:

    def __init__(
        self,
        config: ConfigDict,
        in_dist_dataset_name: str,
        out_of_dist_dataset_name: str,
        log_dir: str,
    ):

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
        self.id_test_dataloader, self.ood_test_dataloader = self._create_dataloaders()
    
    def _initialize_model(self, finetuned: bool = True) -> torch.nn.Module:
        summary = self.config.summary
        n_class = self.config.dataset.n_class

        if self.config.model.pretrained:
            if finetuned:
                print('initialize finetuned model...')
                model = torch.hub.load(
                    repo_or_dir=self.config.model.repo,
                    model=self.config.model.pretrained_model,
                    pretrained=False,
                )
                model.head = nn.Linear(model.head.in_features, n_class)
            else:
                print('initialize pretrained-only model...')
                model = torch.hub.load(
                    repo_or_dir=self.config.model.repo,
                    model=self.config.model.pretrained_model,
                    pretrained=True,
                )
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

        if finetuned:
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
        id_test_dataset = self._create_dataset(self.in_dist_dataset_name, train=False)
        id_test_dataloader = DataLoader(
            dataset=id_test_dataset,
            batch_size=self.config.eval.batch_size,
            shuffle=False,
            num_workers=8,
        )
        ood_test_dataset = self._create_dataset(self.out_of_dist_dataset_name, train=False)
        ood_test_dataloader = DataLoader(
            dataset=ood_test_dataset,
            batch_size=self.config.eval.batch_size,
            shuffle=False,
            num_workers=8,
        )
        return id_test_dataloader, ood_test_dataloader
    
    def _init_metric(self, metric_name, **masking_kwargs):
        id_train_dataset = self._create_dataset(self.in_dist_dataset_name, train=True)
        id_train_dataloader = DataLoader(
            dataset=id_train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=8,
        )
        if metric_name == 'MSP':
            metric = MSP(self.config, self.model)
        elif metric_name == 'Mahalanobis':
            metric = Mahalanobis(self.config, self.model, id_train_dataloader, self.feature_extractor)
        elif metric_name == 'ClasswiseMahalanobis':
            metric = ClasswiseMahalanobis(self.config, self.model, id_train_dataloader, self.feature_extractor)
        elif metric_name == 'SML':
            metric = SML(self.config, self.model, id_train_dataloader)
        elif metric_name == 'DML':
            metric = DML(self.config, self.model, **masking_kwargs)
        elif metric_name == 'DCL':
            metric = DCL(self.config, self.model, **masking_kwargs)
        elif metric_name == 'DMD':
            metric = DMD(self.config, self.model, id_train_dataloader, self.feature_extractor, **masking_kwargs)
        elif metric_name == 'DCMD':
            metric = DCMD(self.config, self.model, id_train_dataloader, self.feature_extractor, **masking_kwargs)
        else:
            raise NotImplementedError('metric not supported')
        
        return metric

    def compute_ood_classification_results(
        self, 
        metric_name: str,
        finetuned: bool,
        **mask_kwargs,
    ):
        print(f'compute ood scores by {metric_name}...')
        self.model = self._initialize_model(finetuned=finetuned)
        self.feature_extractor = FeatureExtractor(
            model=self.model,
            layer_name=self.config.model.layer_name.penultimate,
        )
        self.feature_extractor.hook()

        metric = self._init_metric(metric_name, **mask_kwargs)
        test_y, ood_scores, id_ood_scores, ood_ood_scores = compute_ood_scores(
            metric=metric,
            in_dist_dataloader=self.id_test_dataloader,
            out_of_dist_dataloader=self.ood_test_dataloader,
        )
        _, _, auroc_score = auroc(test_y, ood_scores)
        _, _, aupr_score = aupr(test_y, ood_scores)
        fpr95 = fpr_at_95_tpr(test_y, ood_scores)
        
        return auroc_score, aupr_score, fpr95