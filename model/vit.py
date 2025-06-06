import torch
import torch.nn as nn
import torch.nn.functional as F

# config numbers
# class ViTConfig:
#     def __init__(self,
#                  img_size=32,
#                  patch_size=4,
#                  in_channels=3,
#                  emb_size=64,
#                  depth=6,
#                  num_heads=4,
#                  mlp_ratio=4.0,
#                  num_classes=10,
#                  dropout=0.1):
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.in_channels = in_channels
#         self.emb_size = emb_size
#         self.depth = depth
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.num_classes = num_classes
#         self.dropout = dropout

# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)        # (B, emb_size, H/patch, W/patch)
        x = x.flatten(2)              # (B, emb_size, N)
        x = x.transpose(1, 2)         # (B, N, emb_size)
        return x                      # Shape: (B, N, E)

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, int(emb_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(emb_size * mlp_ratio), emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

# Vit Test Model
class VisionTransformerTest(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["model"]
        
        self.CHANNEL = cfg["in_channels"]
        self.PATCH = cfg["patch_size"]
        self.EMBEDDING = cfg["emb_size"]
        self.IMAGE = cfg["img_size"]
        self.NUM_HEADS = cfg["num_heads"]
        self.MLP_RATIO = cfg["mlp_ratio"]
        self.DROPOUT = cfg["dropout"]
        self.NUM_CLASS = cfg["num_classes"]
        self.DEPTH = cfg["depth"]

        self.patch_embed = PatchEmbedding(
            in_channels=self.CHANNEL,
            patch_size=self.PATCH,
            emb_size=self.EMBEDDING,
            img_size=self.IMAGE
        )

        self.n_patches = (self.IMAGE// self.PATCH) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.EMBEDDING))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, self.EMBEDDING))

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                emb_size=self.EMBEDDING,
                num_heads=self.NUM_HEADS,
                mlp_ratio=self.MLP_RATIO,
                dropout=self.DROPOUT
            ) for _ in range(self.DEPTH)
        ])

        self.norm = nn.LayerNorm(self.EMBEDDING)
        self.head = nn.Linear(self.EMBEDDING, self.NUM_CLASS)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)
