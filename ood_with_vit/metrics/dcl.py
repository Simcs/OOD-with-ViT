from __future__ import annotations

from typing import List
from PIL import Image

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ml_collections.config_dict import ConfigDict

from ood_with_vit.utils import compute_logits
from . import MaskMetric


class DCL(MaskMetric):
    """
    Implementation of Difference of Class Logit metric.
    """
    
    def __init__(
        self, 
        config: ConfigDict,
        model: torch.nn.Module,
        mask_method: str = 'top_ratio',
        mask_ratio: float = 0.3,
        mask_threshold: float = 0.9,
        head_fusion: str = 'max',
        discard_ratio: float = 0.9,
    ):
        super().__init__(
            config=config, 
            model=model, 
            mask_method=mask_method, 
            mask_ratio=mask_ratio, 
            mask_threshold=mask_threshold, 
            head_fusion=head_fusion, 
            discard_ratio=discard_ratio,
        )
        self.attention_masking.hook()
        
    def compute_img_ood_score(self, img: np.ndarray) -> float:
        """
        Compute DML based out-of-distrbution score given a test data.
        """
        self.model.eval()
        with torch.no_grad():
            self.attention_masking.switch_status('normal')
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            original_logit = compute_logits(self.config, self.model, img.unsqueeze(0))
            original_max_logit, original_pred = original_logit.max(dim=1)

            self.attention_masking.switch_status('masking')
            self.attention_masking.generate_mask(img)
            masked_logit = compute_logits(self.config, self.model, img.unsqueeze(0))

        original_max_logit, original_pred = original_max_logit.item(), original_pred.item()
        masked_class_logit = masked_logit.squeeze()[original_pred].item()
        return -(original_max_logit - masked_class_logit)
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            total_dcl = []
            for x, y in tqdm(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.attention_masking.switch_status('normal')
                original_logits = compute_logits(self.config, self.model, x)
                original_max_logits, original_preds = original_logits.max(dim=1)

                self.attention_masking.generate_masks(x)
                self.attention_masking.switch_status('masking')
                masked_logits = compute_logits(self.config, self.model, x)

                for i, (ori_max_logit, ori_pred) in enumerate(zip(original_max_logits, original_preds)):
                    ori_max_logit, ori_pred = ori_max_logit.item(), ori_pred.item()
                    cls_masked_logit = masked_logits[i, ori_pred].item()
                    total_dcl.append(-(ori_max_logit - cls_masked_logit))
        return total_dcl