import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, dim_embed, num_channel=3):
        super().__init__()
        # patch embedding: [B, C, H, W] -> [B, PPC, H/P, W/P]
        self.proj = nn.Conv2d(num_channel, dim_embed, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # flatten: [B, PPC, H/P, W/P] -> [B, PPC, HW/PP(num_patches)]
        # permute: [B, PPC, num_patches] -> [B, num_patches, PPC]
        x = self.proj(x).flatten(2)
        x = x.permute(0, 2, 1)
        return x


class ClsAndPosEmbed(nn.Module):
    def __init__(self, num_patches, dim_embed, drop_ratio=0., cls=True):
        super().__init__()
        self.cls_switch = cls
        if cls:
          self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_embed))
          self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim_embed))
        else:
          self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim_embed))
        self.pos_drop = nn.Dropout(p=drop_ratio)

    def forward(self, x):
        if self.cls_switch:
          cls_token_batch = self.cls_token.expand(x.shape[0], -1, -1)
          x = torch.cat([cls_token_batch, x], dim=1)
        x = torch.add(x, self.pos_embed.to(x.dtype))
        x = self.pos_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, dim_hidden=None, dim_out=None, drop_ratio=0.):
        super().__init__()
        dim_hidden = dim_hidden or dim
        dim_out = dim_out or dim
        self.network = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.Dropout(drop_ratio),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_out),
            nn.Dropout(drop_ratio)
        )

    def forward(self, x):
        x = self.network(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=drop_ratio, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim_hidden=dim * 4, drop_ratio=drop_ratio)

    def attention(self, x):
        b, n, c = x.shape
        # qkv(): [B, num_patches + 1, 3 * total_embed_dim]
        # reshape: [B, num_patches + 1, 3, total_embed_dim]
        # permute: [3, B, num_patches + 1, B, total_embed_dim]
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return (self.mha(q, k, v))[0]

    def forward(self, x):
        x = torch.add(x, self.attention(self.ln1(x)))
        x = torch.add(x, self.mlp(self.ln2(x)))
        return x


class Vit(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, num_att_layer, num_heads, cls=True, num_channel=3, drop_ratio=0.):
        super().__init__()
        assert img_size % patch_size == 0, 'image size must be divisible by patch size'
        num_patches = (img_size // patch_size) ** 2
        dim_embed = num_channel * patch_size * patch_size
        assert dim_embed % num_heads == 0, 'dimension of token must be divisible by number of heads'

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, dim_embed=dim_embed)
        self.cls_switch = cls
        self.cls_pos_embed = ClsAndPosEmbed(num_patches, dim_embed, drop_ratio, cls)
        self.attention = nn.Sequential(
            *[AttentionBlock(dim_embed, num_heads, drop_ratio)
              for _ in range(num_att_layer)]
        )
        self.ln1 = nn.LayerNorm(dim_embed)
        # self.linear1 = nn.Linear(dim_embed,  dim_embed)
        # self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(dim_embed, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cls_pos_embed(x)
        x = self.attention(x)
        if self.cls_switch:
          x = x.select(dim=1, index=0)
        else:
          x = x.mean(dim=1)
        x = self.ln1(x)
        # x = self.linear1(x)
        # x = self.tanh(x)
        x = self.linear2(x)
        return x