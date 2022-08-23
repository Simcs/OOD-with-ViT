from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path
from tqdm import tqdm
import csv

dataset_root = Path('./data')
cifar10_root = dataset_root / 'cifar10'
cifar10_root.mkdir(parents=True, exist_ok=True)
cifar100_root = dataset_root / 'cifar100'
cifar100_root.mkdir(parents=True, exist_ok=True)


def save_train_and_test(path_root, data_cls):
    train_root = path_root / 'train'
    train_root.mkdir(parents=True, exist_ok=True)
    save_into_img(
        train_root, 
        data_cls(
            root=Path('./data'), 
            train=True, 
            download=False, 
    ))

    test_root = path_root / 'test'
    test_root.mkdir(parents=True, exist_ok=True)
    save_into_img(
        test_root, 
        data_cls(
            root=Path('./data'), 
            train=False, 
            download=False, 
    ))


def save_into_img(pathTo, dataset):
    with open(pathTo / '_class_map.tsv', 'w') as f:
        f.write('idx\timg_filename\ttarget\tlabel_name\n')
        for idx, (img, target) in enumerate(tqdm(dataset)):
            img_filename = f'{idx:05d}.jpg'
            img.save(pathTo / img_filename)
            label_name = dataset.classes[target]
            f.write(f'{idx}\t{img_filename}\t{target}\t{label_name}\n')


save_train_and_test(cifar10_root, CIFAR10)

save_train_and_test(cifar100_root, CIFAR100)