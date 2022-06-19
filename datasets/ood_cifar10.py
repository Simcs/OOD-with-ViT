from typing import Callable, List, Optional

from PIL import Image
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms


class OOD_CIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(
        self,
        root: str,
        in_distribution_class_indices: List[int],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        
        self.img_dict = self._images_by_class_index()
        
        self.in_distribution_class_indices = in_distribution_class_indices
        ood_data, ood_targets = [], []
        id_data, id_targets = [], []
        for class_idx in in_distribution_class_indices:
            oc_imgs = self.img_dict[class_idx]
            id_data += oc_imgs
            id_targets += [class_idx for _ in range(len(oc_imgs))]
            
        # for img, target in zip(self.data, self.targets):
        #     if target in self.in_distribution_class_indices:
        #         oc_data.append(img)
                # oc_targets.append(target)
        
        self.data, self.targets = id_data, id_targets
        
    def _images_by_class_index(self):
        img_dict = {}
        for img, target in zip(self.data, self.targets):
            if target not in img_dict:
                img_dict[target] = []
            img_dict[target].append(img)
        return img_dict
    
    def get_transformed_image(self, class_idx: int, idx: int):
        img = self.img_dict[class_idx][idx]
        
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        
        return img
        
            
        
if __name__ == '__main__':
    
    size = 32

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_oc_cifar10 = OOD_CIFAR10(root='./data', in_distribution_class_indices=[i for i in range(9)], train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_oc_cifar10, batch_size=512, shuffle=True, num_workers=8)
    
    test_oc_cifar10 = OOD_CIFAR10(root='./data', in_distribution_class_indices=[i for i in range(9)], train=False, transform=transform_train)
    testloader = torch.utils.data.DataLoader(test_oc_cifar10, batch_size=512, shuffle=True, num_workers=8)
    
    print('Enumerate trainloader...')
    for b_idx, (inputs, targets) in enumerate(trainloader):
        print(f'{b_idx}: {len(inputs)}, {len(targets)}, {targets[0]}')
    print('Enumerate testloader...')
    for b_idx, (inputs, targets) in enumerate(testloader):
        print(f'{b_idx}: {len(inputs)}, {len(targets)}, {targets[0]}')
        
    print(len(train_oc_cifar10))
    print(train_oc_cifar10.class_to_idx)
        