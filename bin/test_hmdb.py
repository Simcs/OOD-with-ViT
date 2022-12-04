import os
import pickle
from torchvision.datasets import HMDB51


with open('/home/simc/workspace/OOD-with-ViT/data/hmdb51/hmdb51_test_metadata.pkl', 'rb') as f:
    hmdb51_test_metadata = pickle.load(f)

dataset_root = '~/workspace/dataset/hmdb51'
dataset_root = os.path.expanduser(dataset_root)
annotation_path = '~/workspace/dataset/testTrainMulti_7030_splits'
annotation_path = os.path.expanduser(annotation_path)


hmdb51_thwc = HMDB51(
    root=dataset_root,
    annotation_path=annotation_path,
    frames_per_clip=16,
    step_between_clips=1,
    frame_rate=2,
    fold=1,
    train=False,
    num_workers=8,
    output_format="THWC",
    _precomputed_metadata=hmdb51_test_metadata,
)
video, audio, class_index = hmdb51_thwc[0]
print(video.shape)

hmdb51_tchw = HMDB51(
    root=dataset_root,
    annotation_path=annotation_path,
    frames_per_clip=16,
    step_between_clips=1,
    frame_rate=2,
    fold=1,
    train=False,
    num_workers=8,
    output_format="TCHW",
    _precomputed_metadata=hmdb51_test_metadata,
)
hmdb51_tchw.video_clips.output_format = "TCHW"
video, audio, class_index = hmdb51_tchw[0]
print(video.shape)

# print(hmdb51_val_ds.full_video_clips.output_format)
# print(hmdb51_val_ds.video_clips.output_format)