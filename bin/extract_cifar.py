from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path
from tqdm import tqdm
import csv

dataset_root = Path('./data')
cifar10_root = dataset_root / 'cifar10'
cifar10_root.mkdir(parents=True, exist_ok=True)
cifar100_root = dataset_root / 'cifar100'
cifar100_root.mkdir(parents=True, exist_ok=True)


def saveTrainAndTest(pathRoot, dataCls):
    train_root = pathRoot / 'train'
    train_root.mkdir(parents=True, exist_ok=True)
    saveIntoImg(
        train_root, 
        dataCls(
            root=Path('./data'), 
            train=True, 
            download=False, 
    ))

    test_root = pathRoot / 'test'
    test_root.mkdir(parents=True, exist_ok=True)
    saveIntoImg(
        test_root, 
        dataCls(
            root=Path('./data'), 
            train=False, 
            download=False, 
    ))


def saveIntoImg(pathTo, dataset):
    with open(pathTo / '_class_map.tsv', 'w') as f:
        f.write('idx\timg_filename\ttarget\tlabel_name\n')
        for idx, (img, target) in enumerate(tqdm(dataset)):
            img_filename = f'{idx:05d}.jpg'
            img.save(pathTo / img_filename)
            label_name = dataset.classes[target]
            f.write(f'{idx}\t{img_filename}\t{target}\t{label_name}\n')

# cifar10 = CIFAR10(
#     root=dataset_root, 
#     train=True, 
#     download=False, 
# )
# cifar10 = CIFAR10(
#     root=dataset_root, 
#     train=False, 
#     download=False, 
# )

# saveTrainAndTest(cifar10_root, CIFAR10)

saveTrainAndTest(cifar100_root, CIFAR100)

# cifar10_train_root = cifar10_root / 'train'
# cifar10_train_root.mkdir(parents=True, exist_ok=True)
# cifar10 = CIFAR10(
#     root=dataset_root, 
#     train=True, 
#     download=False, 
# )
# with open(cifar10_train_root / '_class_map.tsv', 'w') as f:
#     f.write('idx\ttarget\tlabel_name\timg_filename\n')
#     for idx, (img, target) in enumerate(tqdm(cifar10)):
#         img_filename = f'{idx:05d}.jpg'
#         img.save(cifar10_train_root / img_filename)
#         label_name = cifar10.classes[target]
#         f.write(f'{idx}\t{target}\t{label_name}\t{img_filename}\n')

# cifar10_test_root = cifar10_root / 'test'
# cifar10_test_root.mkdir(parents=True, exist_ok=True)
# cifar10 = CIFAR10(
#     root=dataset_root, 
#     train=False, 
#     download=False, 
# )
# with open(cifar10_test_root / '_class_map.tsv', 'w') as f:
#     for idx, (img, target) in enumerate(tqdm(cifar10)):
#         img_filename = f'{idx:05d}.jpg'
#         img.save(cifar10_test_root / img_filename)
#         label_name = cifar10.classes[target]
#         f.write(f'{idx}\t{target}\t{label_name}\n')

# cifar100 = CIFAR100(
#     root=dataset_root, 
#     train=False, 
#     download=False, 
# )
# for idx, (img, target) in enumerate(cifar100):
#     img_filename = f'{idx:05d}.jpg'
#     img.save(cifar100_root / img_filename)

