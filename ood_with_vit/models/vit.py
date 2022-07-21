from typing import Tuple, Union

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, 
                 dim: int, 
                 heads: int = 8, 
                 dim_head: int = 64, 
                 dropout: float = 0.,
                 visualize: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.visualize = visualize

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn_ = attn if self.visualize else None

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out, attn_
    

class Block(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int,
                 dim_head: int,
                 mlp_dim: int,
                 dropout: float = 0.,
                 visualize: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            visualize=visualize
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, weights = self.attn(x)
        x = x + h
        x = self.mlp(x) + x
        return x, weights


class Transformer(nn.Module):
    def __init__(self, 
                 dim: int, 
                 depth: int, 
                 heads: int, 
                 dim_head: int, 
                 mlp_dim: int, 
                 dropout: float = 0.,
                 visualize: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                visualize=visualize,
            ))
            # self.layers.append(nn.ModuleList([
            #     PreNorm(dim, Attention(dim, 
            #                            heads=heads, 
            #                            dim_head=dim_head, 
            #                            dropout=dropout, 
            #                            visualize=visualize)),
            #     PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            # ]))
            
    def forward(self, x):
        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x)
            # h = x
            # x, weights = attn(x)
            # x = x + h
            # x = ff(x) + x
            attn_weights.append(weights)
        return x, attn_weights


class ViT(nn.Module):
    def __init__(self, 
                 *, 
                 image_size: Union[int, Tuple], 
                 patch_size: Union[int, Tuple],
                 num_classes: int,
                 dim: int,
                 depth: int,
                 heads: int,
                 mlp_dim: int, 
                 pool: str = 'cls', 
                 channels: int = 3, 
                 dim_head: int = 64, 
                 dropout: float = 0., 
                 emb_dropout: float = 0.,
                 visualize: bool = False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, visualize)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, _attn_weights = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        out = self.mlp_head(x)
        
        attn_weights = []
        for attn in _attn_weights:
            attn_weights.append(attn.detach().cpu())
        return out, attn_weights
    
    def get_penultimate_features(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, _ = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        penultimate = x
        out = self.mlp_head(x)
        
        return out, penultimate.detach().cpu()