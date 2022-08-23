import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.features = []
        self.handles = []

    def get_features(self, module, input, output):
        self.features.append(output.detach().cpu())

    def hook(self):
        for name, module in self.model.named_modules():
            if self.layer_name in name:
                handle = module.register_forward_hook(self.get_features)
                self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
    def __call__(self, input):
        self.features = []
        with torch.no_grad():
            output = self.model(input)
        return output