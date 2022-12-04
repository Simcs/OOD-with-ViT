from torchvision.datasets import HMDB51

from typing import Tuple

from torch import Tensor


# audio-free version of kinetics for easy collation
class VideoOnlyHMDB51(HMDB51):

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        # print('full:', self.full_video_clips.cumulative_sizes)
        # print('indices:', self.indices)
        # print('sub:', self.video_clips)
        # print(self.video_clips.cumulative_sizes)
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]
        
        # comment out on torchvision 0.13.1
        # if not self._legacy:
        #     # [T,H,W,C] --> [T,C,H,W]
        #     video = video.permute(0, 3, 1, 2)
        # label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, class_index