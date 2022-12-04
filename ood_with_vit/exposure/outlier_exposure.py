import torch
import torch.nn as nn
import torch.nn.functional as F

class OutlierExposure(nn.Module):
    
    def __init__(
        self,
        id: str,
        ood: str,
    ):
        super(OutlierExposure, self).__init__()
        if ood == 'k600':
            n_classes = 613
        elif ood == 'k700-2020':
            n_classes = 680
        assert ood in ['k600', 'k700-2020']
            
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, n_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        