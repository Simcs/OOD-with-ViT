from typing import List
import cv2
import torch

import numpy as np
 

class ViTAttentionRollout:
    def __init__(self,
                 head_fusion: str = 'max',
                 discard_ratio: float = 0.9):
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
    def rollout(self, attentions: List[torch.Tensor]):
        result = torch.eye(attentions[0].size(-1)).unsqueeze(0)
        with torch.no_grad():
            # attention.size() == [1, num_heads, num_patches + 1, num_patches + 1]
            # fused_attention_heads.size() == [1, num_patches + 1, num_patches + 1]
            for attention in attentions:
                if self.head_fusion == 'mean':
                    fused_attention_heads = attention.mean(axis=1)
                elif self.head_fusion == 'max':
                    fused_attention_heads = attention.max(axis=1)[0]
                elif self.head_fusion == 'min':
                    fused_attention_heads = attention.min(axis=1)[0]
                else:
                    raise ValueError('Invalid attention head fusion type.')
        
                # flat.size() == [1, (num_patches + 1) * (num_patches + 1)]
                flat = fused_attention_heads.view(fused_attention_heads.size(0), -1)
                # Drop the lowest attentions
                _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), dim=-1, largest=False)
                # Do not drop the class token
                indices = indices[indices != 0]
                flat[0, indices] = 0
                
                I = torch.eye(fused_attention_heads.size(-1))
                a = (fused_attention_heads + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                
                result = torch.matmul(a, result)
        
        # Look at the total attention between the class token(CLS) and the image patches
        # mask.size() == [num_patches]
        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask

    def get_visualized_masks(self, img, mask):
        img_h, img_w, _ = img.shape
        mask = cv2.resize(mask, (img_h, img_w), interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
    
    