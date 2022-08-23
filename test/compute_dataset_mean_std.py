from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np

svhn = SVHN(
    root='./data', 
    split='train', 
    download=True, 
)
print(type(svhn.labels[0]))

# img_r = np.array([img[0, :, :] / 255. for img in svhn.data])
# img_g = np.array([img[1, :, :] / 255. for img in svhn.data])
# img_b = np.array([img[2, :, :] / 255. for img in svhn.data])
# r_mean, g_mean, b_mean = img_r.mean(), img_g.mean(), img_b.mean()
# r_std, g_std, b_std = img_r.std(), img_g.std(), img_b.std()

# print('svhn mean:', (r_mean, g_mean, b_mean))
# print('svhn std:', (r_std, g_std, b_std))

# mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])