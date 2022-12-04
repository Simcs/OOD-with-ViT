import os
import argparse
import pickle

from torchvision.datasets.kinetics import Kinetics


def prepare_metadata(args):
    dataset, split, force = args.dataset, args.split, args.force
    print(f'preparing {dataset} split {split}...')
    metadata_filename = f'./data/kinetics/{dataset}_{split}_metadata.pkl'
    # metadata_filename = f'./data/kinetics/kinetics400_{split}_metadata.pkl'
    
    if not force and os.path.exists(metadata_filename):
        with open(metadata_filename, 'rb') as f:
            metadata = pickle.load(f)
            return metadata
        
    dataset_root = f'~/workspace/dataset/kinetics/{dataset}'
    dataset_root = os.path.expanduser(dataset_root)

    kinetics_ds = Kinetics(
        root=dataset_root,
        frames_per_clip=16,
        split=split,
        num_workers=16,
        frame_rate=2,
    )
    
    with open(metadata_filename, 'wb') as f:
        pickle.dump(kinetics_ds.metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['k400', 'k600', 'k700-2020'])
    parser.add_argument('--split', choices=['train', 'val', 'all'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    
    if args.split == 'all':
        prepare_metadata(args)
        prepare_metadata(args)
    else:
        prepare_metadata(args)