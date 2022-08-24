import os
import argparse
import pickle

from torchvision.datasets.kinetics import Kinetics


def prepare_metadata(split='train', force=False):
    print(f'preparing kinetics split {split}...')
    metadata_filename = f'./data/kinetics400_{split}_metadata.pkl'
    
    if not force and os.path.exists(metadata_filename):
        with open(metadata_filename, 'rb') as f:
            metadata = pickle.load(f)
            return metadata
        
    dataset_root = '~/workspace/dataset/kinetics/k400'
    dataset_root = os.path.expanduser(dataset_root)

    kinetics400_ds = Kinetics(
        root=dataset_root,
        frames_per_clip=16,
        split=split,
        num_workers=16,
        frame_rate=2,
    )
    
    with open(metadata_filename, 'wb') as f:
        pickle.dump(kinetics400_ds.metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'all'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    
    if args.split == 'all':
        prepare_metadata(split='train', force=force)
        prepare_metadata(split='val', force=force)
    else:
        prepare_metadata(split=args.split, force=force)