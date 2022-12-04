import os
import argparse
import pickle

from torchvision.datasets import HMDB51


def prepare_metadata(split='train', force=False):
    print(f'preparing hmdb split {split}...')
    metadata_filename = f'./data/hmdb51/hmdb51_{split}_metadata.pkl'
    
    if not force and os.path.exists(metadata_filename):
        with open(metadata_filename, 'rb') as f:
            metadata = pickle.load(f)
            return metadata
        
    dataset_root = '~/workspace/dataset/hmdb51'
    dataset_root = os.path.expanduser(dataset_root)
    annotation_path = '~/workspace/dataset/testTrainMulti_7030_splits'
    annotation_path = os.path.expanduser(annotation_path)
    train = True if split == 'train' else False

    hmdb51_ds = HMDB51(
        root=dataset_root,
        annotation_path=annotation_path,
        frames_per_clip=16,
        step_between_clips=1,
        frame_rate=2,
        train=train,
        num_workers=16,
        output_format='TCHW'
    )
    
    with open(metadata_filename, 'wb') as f:
        pickle.dump(hmdb51_ds.metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['train', 'test', 'all'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    
    if args.split == 'all':
        prepare_metadata(split='train', force=force)
        prepare_metadata(split='test', force=force)
    else:
        meta = prepare_metadata(split=args.split, force=force)
