
from typing import List
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ml_collections.config_dict import ConfigDict

from ood_with_vit.utils import compute_logits


class MSP:
    """
    Implementation of Maximum Softmax Probability metric.
    """
    
    def __init__(self, 
                 config: ConfigDict,
                 model: torch.nn.Module):
        self.config = config
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dataset_mean, dataset_std = self.config.dataset.mean, self.config.dataset.std
        img_size = self.config.model.img_size
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
    
    def compute_img_ood_score(self, img: np.ndarray) -> float:
        """
        Compute MSP based out-of-distrbution score given a test img.
        """
        self.model.eval()
        with torch.no_grad():
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            logits = compute_logits(self.config, self.model, img.unsqueeze(0))
            probs = self.softmax(logits)
            msp, _ = probs.max(1)
        return -msp.item()
    
    def compute_dataset_ood_score(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            total_msp = []
            for x, y in tqdm(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                logits = compute_logits(self.config, self.model, x)
                probs = self.softmax(logits)
                msp, _ = probs.max(dim=1)
                total_msp.extend(-msp.numpy())
        
        return total_msp