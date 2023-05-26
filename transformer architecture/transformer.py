# Importing necessary libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL.Image as PImg
import torchvision.transforms as transform
from einops import reduce, rearrange, repeat
from einops.layers.torch import Reduce, Rearrange
from torchsummary import summary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

blackCat = PImg.open('./blackcat.jpg')
blackCatFig = plt.imshow(blackCat)
# plt.show()

# Resize img to 128 * 128 img
imgSize = 128
resize = transform.Compose([transform.Resize((imgSize, imgSize)), transform.ToTensor()])
x = resize(blackCat)
x = x.unsqueeze(0)

# Size for each patch = 16
patchSize = 16
imgPatches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patchSize, s2=patchSize)
# print(imgPatches.size())


class PatchEmbedding(nn.Module):
    def __init__(self, in_c: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = imgSize):
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos = nn.Parameter(torch.randn((img_size//patch_size)**2 + 1, emb_size))
        self.projection = nn.Sequential(
            nn.Conv2d(in_c, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

    def forward(self, patchedImg: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = patchedImg.shape
        patchedImg = self.projection(patchedImg)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        patchedImg = torch.cat([cls_tokens, patchedImg], dim=1)
        patchedImg += self.pos
        return patchedImg


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 384, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.attentionDropout = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        e = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            e.mask_fill(~mask, fill_value)

        scale = self.emb_size ** 0.5
        attention = F.softmax(e, dim=-1) / scale
        attention = self.attentionDropout(attention)
        # sum up over the third axis
        attendedPatch = torch.einsum('bhal, bhlv -> bhav ', attention, values)
        attendedPatch = rearrange(attendedPatch, "b h n d -> b n (h d)")
        attendedPatch = self.proj(attendedPatch)
        return attendedPatch


class AddResidue(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForward(nn.Sequential):
    def __init__(self, emb_size: int, exp: int = 3, drop_out: float = 0):
        super(FeedForward, self).__init__(
            nn.Linear(emb_size, exp * emb_size),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(exp * emb_size, emb_size)
        )


class TransformerBlock(nn.Sequential):
    def __init__(self, emb_size: int = 384, drop_out: float = 0., for_exp: int = 3, for_dropout: float = 0., **kwargs):
        super(TransformerBlock, self).__init__(
            AddResidue(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_out)
            )),
            AddResidue(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForward(emb_size, for_exp, for_dropout, **kwargs),
                nn.Dropout(drop_out)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, transformerDepth: int = 12, **kwargs):
        super().__init__(*[TransformerBlock(**kwargs) for _ in range(transformerDepth)])


class ClassficationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


class ViT(nn.Sequential):
    def __init__(self, in_c=3, patch_size: int = 16, img_size=128, emb_size: int = 384,
                 transformerDepth: int = 15, n_classes: int = 1000, **kwargs):
        super(ViT, self).__init__(
            PatchEmbedding(in_c=in_c, patch_size=patch_size, emb_size=emb_size, img_size=img_size),
            TransformerEncoder(transformerDepth=transformerDepth, emb_size=emb_size, **kwargs),
            ClassficationHead(n_classes=n_classes, emb_size=emb_size)
        )


summary(ViT(), (3, 128, 128), device='cpu')
# embeddedpatches = PatchEmbedding()(x)
# print(TransformerEncoder()(embeddedpatches).shape)
