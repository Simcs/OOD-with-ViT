import os
import argparse
import pickle

from torchvision.datasets import UCF101


def prepare_metadata(split='train', force=False):
    print(f'preparing hmdb split {split}...')
    metadata_filename = f'./data/ucf101/ucf101_{split}_metadata.pkl'
    
    if not force and os.path.exists(metadata_filename):
        with open(metadata_filename, 'rb') as f:
            metadata = pickle.load(f)
            return metadata
        
    dataset_root = '~/workspace/dataset/ucf101'
    dataset_root = os.path.expanduser(dataset_root)
    annotation_path = '~/workspace/dataset/ucfTrainTestlist'
    annotation_path = os.path.expanduser(annotation_path)
    train = True if split == 'train' else False

    ucf101_ds = UCF101(
        root=dataset_root,
        annotation_path=annotation_path,
        frames_per_clip=16,
        step_between_clips=1,
        frame_rate=2,
        train=train,
        num_workers=16,
        output_format='TCHW'
    )
    ucf101_ds.video_clips.output_format = 'TCHW'
    
    with open(metadata_filename, 'wb') as f:
        pickle.dump(ucf101_ds.metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'test', 'all'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force
    
    if args.split == 'all':
        prepare_metadata(split='train', force=force)
        prepare_metadata(split='test', force=force)
    else:
        meta = prepare_metadata(split=args.split, force=force)