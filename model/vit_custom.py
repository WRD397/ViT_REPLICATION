import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    
    def __init__(self, img_size, patch_size, embed_dim, in_channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )  # [B, C, H, W] -> [B, embed_dim, H//P, W//P]
 
    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)  # -> [B, embed_dim, H//P, W//P]
        x = x.flatten(2).transpose(1, 2)  # -> [B, num_patches, embed_dim]
        return x

# building custom attention instead of using the existing multiheadattention method in torch
class Attention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Embedding dim must be divisible by number of heads"

        self.scale = self.head_dim ** -0.5  # Scaling factor: 1/sqrt(dk)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Fused Q, K, V
        self.attn_drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(dim, dim)  # Final projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape

        # Step 1: Linear layer to get Q, K, V
        qkv = self.qkv(x)                      # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]       # [B, num_heads, N, head_dim]

        # Step 2: Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Step 3: Weighted sum of values
        out = (attn @ v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.projection(out)
        out = self.proj_drop(out)

        return out


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path_method(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path_method(x, self.drop_prob, self.training)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_ratio, dropout, drop_path_rate, layerscale_eps=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = Attention(dim=emb_size, num_heads=num_heads)
        self.drop_path1 = DropPath(drop_path_rate)
        self.gamma1 = nn.Parameter(torch.ones(emb_size) * layerscale_eps)
        self.ln2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, int(emb_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(emb_size * mlp_ratio), emb_size),
            nn.Dropout(dropout)
        )
        self.drop_path2 = DropPath(drop_path_rate)
        self.gamma2 = nn.Parameter(torch.ones(emb_size) * layerscale_eps)
    def forward(self, x):
        x = x + self.drop_path1(self.gamma1*self.attn(self.ln1(x)))  # Apply drop-path to attention output
        x = x + self.drop_path2(self.gamma2*self.mlp(self.ln2(x)))   # Apply drop-path to MLP output
        return x
    

class VisionTransformerTiny(nn.Module):
    def __init__(self,
                    CHANNEL,
                    PATCH,
                    EMBEDDING,
                    IMAGE,
                    NUM_HEADS,
                    MLP_RATIO,
                    DROPOUT,
                    NUM_CLASSES,
                    DEPTH,
                    QKV_BIAS,
                    ATTN_DROP_RATE,
                    DROP_PATH_RATE
                ):
        args = locals()
        _ = args.pop('self')
        print('\n********* Model Params *********')
        print(args)
        print('\n')
        super().__init__()
        self.in_channels = CHANNEL
        self.patch_size = PATCH
        self.embed_dim = EMBEDDING
        self.img_size = IMAGE
        self.num_heads = NUM_HEADS
        self.mlp_ratio = MLP_RATIO
        self.dropout = DROPOUT
        self.num_classes = NUM_CLASSES
        self.depth = DEPTH
        self.qkv_bias = QKV_BIAS
        self.attn_drop_rate = ATTN_DROP_RATE
        self.drop_path_rate = DROP_PATH_RATE
        self.patch_embed = PatchEmbedding(
            img_size=IMAGE,
            patch_size=PATCH,
            in_channels=CHANNEL,
            embed_dim=EMBEDDING
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token_dropout = nn.Dropout(p=0.3)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.dropout)

        # Transformer blocks with gradually increasing drop_path
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                emb_size=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                drop_path_rate=dpr[i]
            )
            for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        self.apply(self._init_weights)
        self._init_pos_cls_head()

    def _init_pos_cls_head(self):
        nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-0.02, b=0.02)
        nn.init.trunc_normal_(self.head.weight, mean=0.0, std=0.02, a=-0.02, b=0.02)
        nn.init.constant_(self.head.bias, 0)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        cls_tokens = self.cls_token_dropout(cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, C)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token output

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)