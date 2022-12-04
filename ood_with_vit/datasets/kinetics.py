from torchvision.datasets import Kinetics

from typing import Tuple, Union, List

from torch import Tensor

import csv
import os
import time
import urllib
import warnings
from functools import partial
from multiprocessing import Pool
from os import path
from typing import Any, Callable, Dict, Optional, Tuple, cast

from torch import Tensor

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.vision import VisionDataset

# from .folder import find_classes, make_dataset
# from .utils import download_and_extract_archive, download_url, verify_str_arg, check_integrity
# from .video_utils import VideoClips
# from .vision import VisionDataset

# def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
#     """Checks if a file is an allowed extension.

#     Args:
#         filename (string): path to a file
#         extensions (tuple of strings): extensions to consider (lowercase)

#     Returns:
#         bool: True if the filename ends with one of given extensions
#     """
#     return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


# def make_dataset(
#     directory: str,
#     class_to_idx: Optional[Dict[str, int]] = None,
#     extensions: Optional[Union[str, Tuple[str, ...]]] = None,
#     is_valid_file: Optional[Callable[[str], bool]] = None,
# ) -> List[Tuple[str, int]]:
#     """Generates a list of samples of a form (path_to_sample, class).

#     See :class:`DatasetFolder` for details.

#     Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
#     by default.
#     """
#     directory = os.path.expanduser(directory)
#     print(directory)
#     print(class_to_idx)
#     if class_to_idx is None:
#         _, class_to_idx = find_classes(directory)
#     elif not class_to_idx:
#         raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

#     both_none = extensions is None and is_valid_file is None
#     both_something = extensions is not None and is_valid_file is not None
#     if both_none or both_something:
#         raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

#     if extensions is not None:

#         def is_valid_file(x: str) -> bool:
#             return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

#     is_valid_file = cast(Callable[[str], bool], is_valid_file)

#     from tqdm import tqdm
#     instances = []
#     available_classes = set()
#     for target_class in tqdm(sorted(class_to_idx.keys())):
#         class_index = class_to_idx[target_class]
#         target_dir = os.path.join(directory, target_class)
#         print('target dir:', target_dir)
#         if not os.path.isdir(target_dir):
#             continue
#         for root, _, fnames in os.walk(target_dir, followlinks=True):
#             for fname in sorted(fnames):
#                 # print('fname:', fname)
#                 path = os.path.join(root, fname)
#                 if is_valid_file(path):
#                     item = path, class_index
#                     instances.append(item)

#                     if target_class not in available_classes:
#                         available_classes.add(target_class)

#     empty_classes = set(class_to_idx.keys()) - available_classes
#     if empty_classes:
#         msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
#         if extensions is not None:
#             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
#         raise FileNotFoundError(msg)

#     return instances

# # audio-free version of kinetics for easy collation
# class VideoOnlyKinetics(VisionDataset):
    
#     def __init__(
#         self,
#         root: str,
#         frames_per_clip: int,
#         num_classes: str = "400",
#         split: str = "train",
#         frame_rate: Optional[int] = None,
#         step_between_clips: int = 1,
#         transform: Optional[Callable] = None,
#         extensions: Tuple[str, ...] = ("avi", "mp4"),
#         download: bool = False,
#         num_download_workers: int = 1,
#         num_workers: int = 1,
#         _precomputed_metadata: Optional[Dict[str, Any]] = None,
#         _video_width: int = 0,
#         _video_height: int = 0,
#         _video_min_dimension: int = 0,
#         _audio_samples: int = 0,
#         _audio_channels: int = 0,
#         _legacy: bool = False,
#         output_format: str = "TCHW",
#     ) -> None:

#         # TODO: support test
#         # self.num_classes = verify_str_arg(num_classes, arg="num_classes", valid_values=["400", "600", "700"])
#         self.num_classes = num_classes
#         self.extensions = extensions
#         self.num_download_workers = num_download_workers

#         self.root = root
#         self._legacy = _legacy

#         if _legacy:
#             print("Using legacy structure")
#             self.split_folder = root
#             self.split = "unknown"
#             output_format = "THWC"
#             if download:
#                 raise ValueError("Cannot download the videos using legacy_structure.")
#         else:
#             self.split_folder = path.join(root, split)
#             # self.split = verify_str_arg(split, arg="split", valid_values=["train", "val", "test"])
#             self.split = split

#         if download:
#             self.download_and_process_videos()

#         super().__init__(self.root)

#         print('find classes...')
#         self.classes, class_to_idx = find_classes(self.split_folder)
#         print('make dataset...')
#         self.samples = make_dataset(self.split_folder, class_to_idx, extensions, is_valid_file=None)
#         video_list = [x[0] for x in self.samples]
#         print('create video clips...')
#         self.video_clips = VideoClips(
#             video_list,
#             frames_per_clip,
#             step_between_clips,
#             frame_rate,
#             _precomputed_metadata,
#             num_workers=num_workers,
#             _video_width=_video_width,
#             _video_height=_video_height,
#             _video_min_dimension=_video_min_dimension,
#             _audio_samples=_audio_samples,
#             _audio_channels=_audio_channels,
#             output_format=output_format,
#         )
#         self.transform = transform


# # audio-free version of kinetics for easy collation
class VideoOnlyKinetics(Kinetics):

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        # comment out on torchvision 0.13.1
        # if not self._legacy:
        #     # [T,H,W,C] --> [T,C,H,W]
        #     video = video.permute(0, 3, 1, 2)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label