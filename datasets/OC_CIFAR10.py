from typing import Callable, List, Optional
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

# cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True)
# print(cifar10.class_to_idx)
# print(len(cifar10.targets), type(cifar10.targets[0]))

class OC_CIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(
        self,
        root: str,
        in_distribution_class_indices: List[int],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        
        self.in_distribution_class_indices = in_distribution_class_indices
        oc_data, oc_targets = [], []
        for i in range(len(self.data)):
            img, target = self.data[i], self.targets[i]
            if target in self.in_distribution_class_indices:
                oc_data.append(img)
                oc_targets.append(target)
        
        self.data, self.targets = oc_data, oc_targets
        
if __name__ == '__main__':
    
    size = 32

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_oc_cifar10 = OC_CIFAR10(root='./data', in_distribution_class_indices=[i for i in range(9)], train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_oc_cifar10, batch_size=512, shuffle=True, num_workers=8)
    
    test_oc_cifar10 = OC_CIFAR10(root='./data', in_distribution_class_indices=[i for i in range(9)], train=False, transform=transform_train)
    testloader = torch.utils.data.DataLoader(test_oc_cifar10, batch_size=512, shuffle=True, num_workers=8)
    
    print('Enumerate trainloader...')
    for b_idx, (inputs, targets) in enumerate(trainloader):
        print(f'{b_idx}: {len(inputs)}, {len(targets)}, {targets[0]}')
    print('Enumerate testloader...')
    for b_idx, (inputs, targets) in enumerate(testloader):
        print(f'{b_idx}: {len(inputs)}, {len(targets)}, {targets[0]}')
        
    print(len(train_oc_cifar10))
    print(train_oc_cifar10.class_to_idx)
        