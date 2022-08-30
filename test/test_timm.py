from tkinter import Place
import timm
from pprint import pprint
import torch
from ood_with_vit.datasets.dtd import DTD
from torchvision.datasets import Places365

places365_train_dataset = Places365(
    root='./data',
    split='train-standard',
    download=True,
)
places365_val_dataset = Places365(
    root='./data',
    split='val',
    download=True,
)


# dtd_train_dataset = DTD(
#     root='./data',
#     split='train',
#     download=True,
# )

# dtd_val_dataset = DTD(
#     root='./data',
#     split='val',
#     download=True,
# )

# dtd_test_dataset = DTD(
#     root='./data',
#     split='test',
#     download=True,
# )


ls_models = timm.list_models(pretrained=True)
vit_models = [model for model in ls_models if 'vit' in model]
deit_models = [model for model in ls_models if 'deit' in model]
pprint(ls_models)
# pprint(vit_models)
# pprint(deit_models)
# print(len(vit_models))

# vit_base = timm.create_model(
#     model_name='vit_base_patch16_224', 
#     pretrained=True,
# )
# print(vit_base)

# print('vitbase:', type(vit_base), isinstance(vit_base, torch.nn.Module))

# deit_base = timm.create_model(
#     model_name='deit_base_patch16_224',
#     pretrained=True,
# )
# print(deit_base)