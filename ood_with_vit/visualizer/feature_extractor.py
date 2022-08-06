import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        self.model = model
        self.features = []
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_features)

    def get_features(self, module, input, output):
        self.features.append(output.detach().cpu())
        
    def __call__(self, input):
        self.features = []
        with torch.no_grad():
            output = self.model(input)
        return output