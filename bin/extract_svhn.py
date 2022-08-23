from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from pathlib import Path
from tqdm import tqdm
import csv

dataset_root = Path('./assets')
svhn_root = dataset_root / 'svhn'
svhn_root.mkdir(parents=True, exist_ok=True)


def save_train_and_test(path_root):
    train_root = path_root / 'train'
    train_root.mkdir(parents=True, exist_ok=True)
    save_into_img(
        path_to=train_root, 
        dataset=SVHN(
            root=Path('./data'), 
            split='train', 
            download=False,
        )
    )

    test_root = path_root / 'test'
    test_root.mkdir(parents=True, exist_ok=True)
    save_into_img(
        path_to=test_root, 
        dataset=SVHN(
            root=Path('./data'), 
            split='test', 
            download=False, 
        )
    )

def save_into_img(path_to, dataset):
    with open(path_to / '_class_map.tsv', 'w') as f:
        f.write('idx\timg_filename\ttarget\tlabel_name\n')
        for idx, (img, target) in enumerate(tqdm(dataset)):
            img_filename = f'{idx:05d}.jpg'
            img.save(path_to / img_filename)
            label_name = target
            f.write(f'{idx}\t{img_filename}\t{target}\t{label_name}\n')

save_train_and_test(svhn_root)
