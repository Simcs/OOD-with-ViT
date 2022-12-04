from typing import List, Optional, Callable

import os
import torch
import json
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset, DataLoader


class KineticsOOD(Dataset):
    
    def __init__(
        self,
        mode: str,
        id: str,
        ood: str,
        train: bool = True,
        n_shot: int = 10,
        transform: Optional[Callable] = None,
        head_fusion: Optional[str] = None,
        discard_ratio: Optional[float] = None,
        spatial_masking: Optional[bool] = None,
        spatial_mask_method: Optional[str] = None,
        spatial_mask_ratio: Optional[float] = None,
        spatial_mask_threshold: Optional[float] = None,
        temporal_masking: Optional[bool] = None,
        temporal_mask_method: Optional[str] = None,
        temporal_mask_ratio: Optional[float] = None,
        temporal_mask_threshold: Optional[float] = None,
        _id_embeddings: Optional[List] = None,
        _id_labels: Optional[List] = None,
        _ood_embeddings: Optional[List] = None,
        _ood_labels: Optional[List] = None,
    ):
        self.mode = mode
        self.id = id
        self.ood = ood
        self.transform = transform
        
        self.head_fusion, self.discard_ratio = head_fusion, discard_ratio
        self.spatial_masking = spatial_masking
        self.spatial_mask_method = spatial_mask_method
        self.spatial_mask_ratio = spatial_mask_ratio
        self.spatial_mask_threshold = spatial_mask_threshold
        self.temporal_masking = temporal_masking
        self.temporal_mask_method = temporal_mask_method
        self.temporal_mask_ratio = temporal_mask_ratio
        self.temporal_mask_threshold = temporal_mask_threshold
        
        self.id_embeddings, self.id_labels = None, None
        self.ood_embeddings, self.ood_labels = None, None
        if _id_embeddings is not None:
            self.id_embeddings = _id_embeddings
        if _id_labels is not None:
            self.id_labels = _id_labels
        if _ood_embeddings is not None:
            self.ood_embeddings = _ood_embeddings
        if _ood_labels is not None:
            self.ood_labels = _ood_labels
        if spatial_mask_threshold is not None:
            n_shot = self.get_n_shot(n_shot, spatial_mask_threshold)
        
        # if self.mode == 'mask':
        #     id_masked_emb_fn, ood_masked_emb_fn = self.get_masked_embeddings_filename()
        # # else:
        # id_emb_fn, ood_emb_fn = self.get_original_embeddings_filename()
        
        if self.mode == 'mask':
            id_emb_fn, ood_emb_fn = self.get_masked_embeddings_filename()
        else:
            id_emb_fn, ood_emb_fn = self.get_original_embeddings_filename()
        
        # load in-distribution embeddings
        if self.id_embeddings is None or self.id_labels is None:
            self.id_embeddings, self.id_labels = [], []
            print(f'loading {id_emb_fn}...')
            with open(id_emb_fn) as f:
                for line in tqdm(f):
                    emb_js = json.loads(line)
                    label = emb_js['gt']
                    pre_logit = np.array(emb_js['penultimate'])
                    self.id_embeddings.append(pre_logit)
                    self.id_labels.append(label)
        
        # load out-of-distribution embeddings
        if self.ood_embeddings is None or self.ood_labels is None:
            ood_emb_fn = ood_emb_fn.parent / (ood_emb_fn.stem + '_deduplicated' + ood_emb_fn.suffix)
            self.ood_embeddings, self.ood_labels = [], []
            print(f'loading {ood_emb_fn}...')
            with open(ood_emb_fn, 'r') as f:
                for line in tqdm(f):
                    emb_js = json.loads(line)
                    label = emb_js['gt']
                    pre_logit = np.array(emb_js['penultimate'])
                    self.ood_embeddings.append(pre_logit)
                    self.ood_labels.append(label)
        
        id_embeddings_per_class = {}
        for label, emb in zip(self.id_labels, self.id_embeddings):
            if label not in id_embeddings_per_class:
                id_embeddings_per_class[label] = []
            id_embeddings_per_class[label].append(emb)
        
        ood_embeddings_per_class = {}
        for label, emb in zip(self.ood_labels, self.ood_embeddings):
            if label not in ood_embeddings_per_class:
                ood_embeddings_per_class[label] = []
            ood_embeddings_per_class[label].append(emb)
        
        if train:
            train_id_embeddings, train_id_labels = [], []
            for label in id_embeddings_per_class.keys():
                n = int(len(id_embeddings_per_class[label]) * 0.9)
                id_embeddings_per_class[label] = id_embeddings_per_class[label][:n]
                train_id_embeddings += id_embeddings_per_class[label]
                train_id_labels += [label] * n
            
            train_ood_embeddings, train_ood_labels = [], []
            for label in ood_embeddings_per_class.keys():
                n = int(len(ood_embeddings_per_class[label]) * 0.9)
                ood_embeddings_per_class[label] = ood_embeddings_per_class[label][:n]
                train_ood_embeddings += ood_embeddings_per_class[label]
                train_ood_labels += [label] * n
            
            fs_ood_embeddings, fs_ood_labels = [], []
            n_id_classes, n_ood_classes = len(list(set(train_id_labels))), len(list(set(train_ood_labels)))
            oversampling_factor = len(train_id_embeddings) / n_shot / n_id_classes
            
            for label in ood_embeddings_per_class.keys():
                fs = random.sample(ood_embeddings_per_class[label], n_shot)
                for _ in range(int(oversampling_factor)):
                    fs_ood_embeddings += fs
                    fs_ood_labels += [label] * n_shot
                n_remain = int((oversampling_factor - int(oversampling_factor)) * len(fs))
                fs_ood_embeddings += fs[:n_remain]
                fs_ood_labels += [label] * n_remain
                
            self.embeddings = train_id_embeddings + fs_ood_embeddings
            self.labels = train_id_labels + fs_ood_labels
            print('#id embeddings:', len(train_id_embeddings), '#ood embeddings:', len(fs_ood_embeddings))

        else:
            val_id_embeddings, val_id_labels = [], []
            for label in id_embeddings_per_class.keys():
                n = int(len(id_embeddings_per_class[label]) * 0.9) + 1
                id_embeddings_per_class[label] = id_embeddings_per_class[label][n:]
                val_id_embeddings += id_embeddings_per_class[label]
                val_id_labels += [label] * len(id_embeddings_per_class[label])
            
            val_ood_embeddings, val_ood_labels = [], []
            for label in ood_embeddings_per_class.keys():
                n = int(len(ood_embeddings_per_class[label]) * 0.9) + 1
                ood_embeddings_per_class[label] = ood_embeddings_per_class[label][n:]
                val_ood_embeddings += ood_embeddings_per_class[label]
                val_ood_labels += [label] * len(ood_embeddings_per_class[label])
            
            self.embeddings = val_id_embeddings + val_ood_embeddings
            self.labels = val_id_labels + val_ood_labels
            print('#id embeddings:', len(val_id_embeddings), '#ood embeddings:', len(val_ood_embeddings))
        
                    
        # if self.id_embeddings is None or self.id_labels is None:
        #     self.id_ori_embeddings, self.id_ori_labels = [], []
        #     self.id_masked_embeddings, self.id_masked_labels = [], []
            
        #     print(f'loading {id_emb_fn}...')
        #     with open(id_emb_fn, 'r') as f:
        #         for line in tqdm(f):
        #             emb_js = json.loads(line)
        #             label = emb_js['gt']
        #             pre_logit = np.array(emb_js['penultimate'])
        #             # self.id_embeddings.append(pre_logit)
        #             # self.id_labels.append(label)
        #             self.id_ori_embeddings.append(pre_logit)
        #             self.id_ori_labels.append(label)
                    
        #     self.id_embeddings = self.id_ori_embeddings
        #     self.id_labels = self.id_ori_labels
            
        #     if self.mode == 'mask':
        #         with open(id_masked_emb_fn, 'r') as f:
        #             for line in tqdm(f):
        #                 emb_js = json.loads(line)
        #                 label = emb_js['gt']
        #                 pre_logit = np.array(emb_js['penultimate'])
        #                 # self.id_embeddings.append(pre_logit)
        #                 # self.id_labels.append(label)
        #                 self.id_masked_embeddings.append(pre_logit)
        #                 self.id_masked_labels.append(label)
            
        #         self.id_embeddings = self.id_ori_embeddings + self.id_masked_embeddings
        #         self.id_labels = self.id_ori_labels + self.id_masked_labels
                
        
        # if self.ood_embeddings is None or self.ood_labels is None:
        #     ood_emb_fn = ood_emb_fn.parent / (ood_emb_fn.stem + '_deduplicated' + ood_emb_fn.suffix)
        #     self.ood_ori_embeddings, self.ood_ori_labels = [], []
        #     self.ood_masked_embeddings, self.ood_masked_labels = [], []
        #     print(f'loading {ood_emb_fn}...')
        #     with open(ood_emb_fn, 'r') as f:
        #         for line in tqdm(f):
        #             emb_js = json.loads(line)
        #             label = emb_js['gt']
        #             pre_logit = np.array(emb_js['penultimate'])
        #             self.ood_ori_embeddings.append(pre_logit)
        #             self.ood_ori_labels.append(label)
                    
        #     self.ood_embeddings = self.ood_ori_embeddings
        #     self.ood_labels = self.ood_ori_labels
            
        #     if self.mode == 'mask':
        #         ood_masked_emb_fn = ood_masked_emb_fn.parent / (ood_masked_emb_fn.stem + '_deduplicated' + ood_masked_emb_fn.suffix)
        #         with open(ood_masked_emb_fn, 'r') as f:
        #             for line in tqdm(f):
        #                 emb_js = json.loads(line)
        #                 label = emb_js['gt']
        #                 pre_logit = np.array(emb_js['penultimate'])
        #                 # self.id_embeddings.append(pre_logit)
        #                 # self.id_labels.append(label)
        #                 self.ood_masked_embeddings.append(pre_logit)
        #                 self.ood_masked_labels.append(label)
            
        #         self.ood_embeddings = self.ood_ori_embeddings + self.ood_masked_embeddings
        #         self.ood_labels = self.ood_ori_labels + self.ood_masked_labels
        
        # if train:
        #     fs_ood_embeddings, fs_ood_labels = [], []
        #     ood_embeddings_per_class = {}
            
        
        # if train:
        #     few_shot_labels = []
        #     few_shot_ood_embeddings = []
            
        #     ood_ori_embeddings_per_class = {}
        #     for emb, label in zip(self.ood_ori_embeddings, self.ood_ori_labels):
        #         if label not in ood_ori_embeddings_per_class:
        #             ood_ori_embeddings_per_class[label] = []
        #         ood_ori_embeddings_per_class[label].append(emb)
            
        #     for label, embeddings in ood_ori_embeddings_per_class.items():
        #         few_shot_ood_embeddings.extend(random.sample(embeddings, n_shot))
        #         for _ in range(n_shot):
        #             few_shot_labels.append(label)
                    
        #     if mode == 'mask':
        #         ood_masked_embeddings_per_class = {}
        #         for emb, label in zip(self.ood_masked_embeddings, self.ood_masked_labels):
        #             if label not in ood_masked_embeddings_per_class:
        #                 ood_masked_embeddings_per_class[label] = []
        #             ood_masked_embeddings_per_class[label].append(emb)
                    
        #         for label, embeddings in ood_masked_embeddings_per_class.items():
        #             few_shot_ood_embeddings.extend(random.sample(embeddings, n_shot))
        #             for _ in range(n_shot):
        #                 few_shot_labels.append(label)
            
        # # few_shot_labels = []
        # # few_shot_ood_embeddings = []
        # # for label, embeddings in ood_embeddings_per_class.items():
        # #     few_shot_ood_embeddings.extend(random.sample(embeddings, n_shot))
        # #     for _ in range(n_shot):
        # #         few_shot_labels.append(label)
        
        # if train:
        #     self.embeddings = self.id_embeddings + few_shot_ood_embeddings
        #     self.labels = self.id_labels + few_shot_labels
        # else:
        #     self.embeddings = self.id_embeddings + self.ood_embeddings
        #     self.labels = self.id_labels + self.ood_labels
            
        
        self.classes = list(set(self.labels))
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        
        self.targets = []
        for label in self.labels:
            self.targets.append(self.class_to_idx[label])
        
        print('#id all:', len(self.id_embeddings), '#ood all:', len(self.ood_embeddings))
        print('#id label:', len(set(self.id_labels)), '#ood label:', len(set(self.ood_labels)))        
        print('#targets:', len(self.targets), len(set(self.targets)))
        
    def get_n_shot(self, n, f):
        return int(n * pow(1.1, 7-abs(7-int(1000*f))))
        
    def get_original_embeddings_filename(self):
        id_embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'original'
        id_emb_fn = id_embeddings_dir / f'{self.id}_val_embeddings.jsonl'
        
        ood_embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'original'
        ood_emb_fn = ood_embeddings_dir / f'{self.ood}_val_embeddings.jsonl'
        
        return id_emb_fn, ood_emb_fn
    
    def get_masked_embeddings_filename(self):
        head_fusion, discard_ratio = self.head_fusion, self.discard_ratio
        spatial_masking, temporal_masking = self.spatial_masking, self.temporal_masking
        spatial_mask_method = self.spatial_mask_method
        spatial_mask_ratio, spatial_mask_threshold = self.spatial_mask_ratio, self.spatial_mask_threshold
        temporal_mask_method = self.temporal_mask_method
        temporal_mask_ratio, temporal_mask_threshold = self.temporal_mask_ratio, self.temporal_mask_threshold
    
        id_embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'masked' / f'{self.id}_val'
        id_emb_fn = id_embeddings_dir / f'{head_fusion}_{discard_ratio}'
        if spatial_masking and not temporal_masking:
            if 'ratio' in spatial_mask_method:
                id_emb_fn = id_emb_fn.parent / f'spatial_{id_emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.jsonl'
            else:
                id_emb_fn = id_emb_fn.parent / f'spatial_{id_emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.jsonl'
        elif not spatial_masking and temporal_masking:
            if 'ratio' in temporal_mask_method:
                id_emb_fn = id_emb_fn.parent / f'temporal_{id_emb_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.jsonl'
            else:
                id_emb_fn = id_emb_fn.parent / f'temporal_{id_emb_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.jsonl'
        elif spatial_masking and temporal_masking:
            if 'ratio' in spatial_mask_method:
                if 'ratio' in temporal_mask_method:
                    id_emb_fn = id_emb_fn.parent / (f'spatiotemporal_{id_emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                        f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
                else:
                    id_emb_fn = id_emb_fn.parent / (f'spatiotemporal_{id_emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                        f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
            else:
                if 'ratio' in temporal_mask_method:
                    id_emb_fn = id_emb_fn.parent / (f'spatiotemporal_{id_emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                        f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
                else:
                    id_emb_fn = id_emb_fn.parent / (f'spatiotemporal_{id_emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                        f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
                id_embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'masked' / f'{self.id}_val'
        
        ood_embeddings_dir = Path('./data') / 'kinetics' / 'embeddings' / 'masked' / f'{self.ood}_val'
        ood_emb_fn = ood_embeddings_dir / f'{head_fusion}_{discard_ratio}'
        if spatial_masking and not temporal_masking:
            if 'ratio' in spatial_mask_method:
                ood_emb_fn = ood_emb_fn.parent / f'spatial_{ood_emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}.jsonl'
            else:
                ood_emb_fn = ood_emb_fn.parent / f'spatial_{ood_emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}.jsonl'
        elif not spatial_masking and temporal_masking:
            if 'ratio' in temporal_mask_method:
                ood_emb_fn = ood_emb_fn.parent / f'temporal_{ood_emb_fn.name}_{temporal_mask_method}_{temporal_mask_ratio}.jsonl'
            else:
                ood_emb_fn = ood_emb_fn.parent / f'temporal_{ood_emb_fn.name}_{temporal_mask_method}_{temporal_mask_threshold}.jsonl'
        elif spatial_masking and temporal_masking:
            if 'ratio' in spatial_mask_method:
                if 'ratio' in temporal_mask_method:
                    ood_emb_fn = ood_emb_fn.parent / (f'spatiotemporal_{ood_emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                        f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
                else:
                    ood_emb_fn = ood_emb_fn.parent / (f'spatiotemporal_{ood_emb_fn.name}_{spatial_mask_method}_{spatial_mask_ratio}_' + \
                        f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
            else:
                if 'ratio' in temporal_mask_method:
                    ood_emb_fn = ood_emb_fn.parent / (f'spatiotemporal_{ood_emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                        f'{temporal_mask_method}_{temporal_mask_ratio}.jsonl')
                else:
                    ood_emb_fn = ood_emb_fn.parent / (f'spatiotemporal_{ood_emb_fn.name}_{spatial_mask_method}_{spatial_mask_threshold}_' + \
                        f'{temporal_mask_method}_{temporal_mask_threshold}.jsonl')
        
        return id_emb_fn, ood_emb_fn
    
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, index):
        embedding = torch.Tensor(self.embeddings[index])
        target = self.targets[index]
        return embedding, target