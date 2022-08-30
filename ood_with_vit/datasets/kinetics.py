from torchvision.datasets import Kinetics

from typing import Tuple

from torch import Tensor


# audio-free version of kinetics for easy collation
class MyKinetics(Kinetics):

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        if not self._legacy:
            # [T,H,W,C] --> [T,C,H,W]
            video = video.permute(0, 3, 1, 2)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label