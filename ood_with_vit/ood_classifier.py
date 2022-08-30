from typing import Optional, Tuple
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, LSUN, Places365
from torchvision.datasets import ImageFolder

from ml_collections import ConfigDict
import timm

from ood_with_vit.models.vit import ViT
from ood_with_vit.datasets.dtd import DTD
from ood_with_vit.metrics import MSP, Mahalanobis, ClasswiseMahalanobis, SML
from ood_with_vit.metrics import DCL, DML, DMD, DCMD, MCMD
from ood_with_vit.utils import compute_ood_scores
from ood_with_vit.utils.ood_metrics import auroc, aupr, fpr_at_95_tpr
from ood_with_vit.visualizer.feature_extractor import FeatureExtractor


DATASETS = ['CIFAR10', 'CIFAR100', 'SVHN', 'LSUN', 'TinyImageNet', 'DTD', 'Places365']

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
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])

        # create model and test dataloaders
        self.id_test_dataloader, self.ood_test_dataloader = self._create_dataloaders()

        self.model = None
        self.feature_extractor = None
    
    def _initialize_model(self, finetuned: bool = True) -> torch.nn.Module:
        summary = self.config.summary
        n_class = self.config.dataset.n_class

        if self.config.model.pretrained:
            if finetuned:
                print('initialize finetuned model...')
                model = timm.create_model(
                    model_name=self.config.model.pretrained_model,
                    pretrained=False,
                )
                model.head = nn.Linear(model.head.in_features, n_class)
            else:
                print('initialize pretrained-only model...')
                model = timm.create_model(
                    model_name=self.config.model.pretrained_model,
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
            # epoch = 50
            # checkpoint = torch.load(self.checkpoint_path / f'{summary}_{epoch}.pt')
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
        elif dataset_name == 'SVHN':
            split = 'train' if train else 'test'
            dataset = SVHN(
                root=dataset_root, 
                split=split, 
                download=True, 
                transform=self.transform_test
            )
        elif dataset_name == 'TinyImageNet':
            split = 'train' if train else 'val'
            imagenet_root = f'{dataset_root}/tiny-imagenet-200/{split}'
            dataset = ImageFolder(
                root=imagenet_root,
                transform=self.transform_test,
            )
        elif dataset_name == 'DTD':
            split = 'train' if train else 'val'
            dataset = DTD(
                root=dataset_root,
                split=split,
                download=True,
                transform=self.transform_test,
            )
        elif dataset_name == 'Places365':
            split = 'train-standard' if train else 'val'
            dataset = Places365(
                root=dataset_root,
                split=split,
                download=True,
                transform=self.transform_test,
            )
        else:
            raise ValueError(f'unsupported dataset {dataset_name}')
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
    
    def _init_metric(
        self, 
        metric_name: str, 
        mask_method: str,
        mask_ratio: float,
        mask_threshold: float,
        head_fusion: str = 'max',
        discard_ratio: float = 0.9,
        _precomputed_statistics: Optional[object] = None,
    ):
        # precomputed_statisitcs, **masking_kwargs):
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
            metric = Mahalanobis(
                self.config, 
                self.model, 
                id_train_dataloader, 
                self.feature_extractor,
                _precomputed_statistics,
            )
        elif metric_name == 'ClasswiseMahalanobis':
            metric = ClasswiseMahalanobis(
                self.config, 
                self.model, 
                id_train_dataloader, 
                self.feature_extractor,
                _precomputed_statistics,
            )
        elif metric_name == 'SML':
            metric = SML(
                self.config, 
                self.model, 
                id_train_dataloader,
                _precomputed_statistics,
            )
        elif metric_name == 'DML':
            metric = DML(
                config=self.config, 
                model=self.model, 
                mask_method=mask_method,
                mask_ratio=mask_ratio,
                mask_threshold=mask_threshold,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio,
            )
        elif metric_name == 'DCL':
            metric = DCL(
                config=self.config, 
                model=self.model, 
                mask_method=mask_method,
                mask_ratio=mask_ratio,
                mask_threshold=mask_threshold,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio,
            )
        elif metric_name == 'DMD':
            metric = DMD(
                config=self.config, 
                model=self.model, 
                id_dataloader=id_train_dataloader, 
                feature_extractor=self.feature_extractor, 
                mask_method=mask_method,
                mask_ratio=mask_ratio,
                mask_threshold=mask_threshold,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio,
            )
        elif metric_name == 'DCMD':
            metric = DCMD(
                config=self.config, 
                model=self.model, 
                id_dataloader=id_train_dataloader, 
                feature_extractor=self.feature_extractor, 
                mask_method=mask_method,
                mask_ratio=mask_ratio,
                mask_threshold=mask_threshold,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio,
            )
        elif metric_name == 'MCMD':
            metric = MCMD(
                config=self.config, 
                model=self.model, 
                id_dataloader=id_train_dataloader, 
                feature_extractor=self.feature_extractor, 
                mask_method=mask_method,
                mask_ratio=mask_ratio,
                mask_threshold=mask_threshold,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio,
                _precomputed_statistics=_precomputed_statistics,
            )
        else:
            raise NotImplementedError('metric not supported')
        
        return metric

    def compute_accuracy(self):
        self.model = self._initialize_model(finetuned=True)
        self.criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        total_test_loss, n_correct, n_total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(self.id_test_dataloader)):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_test_loss += loss.item()
                _, predicted = outputs.max(1)
                n_total += y.size(0)
                n_correct += predicted.eq(y).sum().item()
        
            avg_test_loss = total_test_loss / (batch_idx + 1)
            test_accuracy = 100. * n_correct / n_total
            print(f'Test Loss: {avg_test_loss:.3f} | Test Acc: {test_accuracy:.3f}% ({n_correct}/{n_total})')
        
        return total_test_loss, test_accuracy

    def compute_ood_classification_results(
        self, 
        metric_name: str,
        finetuned: bool,
        mask_method: str = 'lt_threshold',
        mask_ratio: float = 0.3,
        mask_threshold: float = 0.1,
        _precomputed_statistics = None,
    ):
        print(f'compute ood scores by {metric_name}...')
        self.model = self._initialize_model(finetuned=finetuned)
        self.feature_extractor = FeatureExtractor(
            model=self.model,
            layer_name=self.config.model.layer_name.penultimate,
        )
        self.feature_extractor.hook()

        self.metric = self._init_metric(
            metric_name=metric_name, 
            mask_method=mask_method,
            mask_ratio=mask_ratio,
            mask_threshold=mask_threshold,
            head_fusion='max',
            discard_ratio=0.5,
            _precomputed_statistics=_precomputed_statistics,
        )
        test_y, ood_scores, id_ood_scores, ood_ood_scores = compute_ood_scores(
            metric=self.metric,
            in_dist_dataloader=self.id_test_dataloader,
            out_of_dist_dataloader=self.ood_test_dataloader,
        )
        _, _, auroc_score = auroc(test_y, ood_scores)
        _, _, aupr_score = aupr(test_y, ood_scores)
        fpr95 = fpr_at_95_tpr(test_y, ood_scores)
        
        return auroc_score, aupr_score, fpr95