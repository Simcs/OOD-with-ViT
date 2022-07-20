
from PIL import Image

import torch
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
    
    def _compute_statistics(self, img):
        """
        Compute maximum softmax probabiWlity
        return: 
        """
        self.model.eval()
        if self.config.model.pretrained:
            logit = self.model(img.unsqueeze(0))
        else:
            logit, _ = self.model(img.unsqueeze(0))
        probs = self.softmax(logit)
        msp, _ = probs.max(1)
        return msp
        
    def compute_ood_score(self, img):
        """
        Compute MSP based out-of-distrbution score given a test img.
        """
        # msp = self._compute_statistics(img).item()
        self.model.eval()
        with torch.no_grad():
            img = self.transform_test(Image.fromarray(img)).to(self.device)
            logits = compute_logits(self.config, self.model, img.unsqueeze(0))
            probs = self.softmax(logits)
            msp, _ = probs.max(1)
        return -msp.item()