"""
VideoMAE Model — ViT-Tiny
Reference: https://github.com/MCG-NJU/VideoMAE

Contains:
- PatchEmbed3D: 3D patch embedding with tubelet
- Block: Transformer block with LayerScale
- PretrainVisionTransformer: encoder + decoder for masked video reconstruction
- VisionTransformerForFinetune: encoder + classification head
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import trunc_normal_


# ─── Positional Encoding ───────────────────────────────────────────────

def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoidal positional encoding table (non-learnable)."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.tensor(sinusoid_table, dtype=torch.float,
                        requires_grad=False).unsqueeze(0)


# ─── Patch Embedding ───────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding using Conv3d.
    Maps video [B, C, T, H, W] → patches [B, N, embed_dim]
    where N = (T/tubelet) * (H/patch) * (W/patch)
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=3,
                 embed_dim=192, num_frames=8, tubelet_size=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames

        self.grid_size = (
            num_frames // tubelet_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)          # [B, embed_dim, T', H', W']
        x = x.flatten(2)          # [B, embed_dim, N]
        x = x.transpose(1, 2)    # [B, N, embed_dim]
        return x


# ─── Transformer Components ───────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention with optional qkv bias (following VideoMAE)."""
    def __init__(self, dim, num_heads=3, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ])

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP block with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class Block(nn.Module):
    """Transformer block with LayerScale (following VideoMAE/BEiT)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 init_values=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # LayerScale
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ─── PretrainVisionTransformer ────────────────────────────────────────

class PretrainVisionTransformerEncoder(nn.Module):
    """
    VideoMAE Encoder: processes only VISIBLE patches (after masking).
    This is the key efficiency of VideoMAE — with 90% masking,
    the encoder only sees 10% of tokens.
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=0.,
                 tubelet_size=2, num_frames=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=num_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # Sinusoidal positional embeddings (non-learnable)
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, T, H, W] video tensor
            mask: [B, N] boolean mask (True = masked/invisible)
        Returns:
            x_vis: [B, N_vis, embed_dim] visible token features
        """
        x = self.patch_embed(x)  # [B, N, C]

        # Add positional embedding
        pos = self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = x + pos

        B, N, C = x.shape
        # Select only visible (unmasked) patches
        x_vis = x[~mask].reshape(B, -1, C)  # [B, N_vis, C]

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis


class PretrainVisionTransformerDecoder(nn.Module):
    """
    VideoMAE Decoder: lightweight transformer that reconstructs masked patches.
    Receives concatenation of encoded visible tokens + mask tokens.
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=96,
                 depth=4, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm,
                 init_values=None, tubelet_size=2):
        super().__init__()
        self.num_classes = num_classes  # = 3 * tubelet_size * patch_size^2
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        """
        Args:
            x: [B, N, embed_dim] all tokens (visible + mask)
            return_token_num: number of mask tokens to predict
        Returns:
            [B, return_token_num, num_classes] pixel predictions
        """
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            # Only predict masked tokens (appended at end)
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x


class PretrainVisionTransformer(nn.Module):
    """
    Complete VideoMAE model for pre-training.
    Encoder processes visible patches, decoder reconstructs masked patches.
    """
    def __init__(self, img_size=128, patch_size=16,
                 encoder_embed_dim=192, encoder_depth=12,
                 encoder_num_heads=3,
                 decoder_embed_dim=96, decoder_depth=4,
                 decoder_num_heads=3,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0., tubelet_size=2, num_frames=8):
        super().__init__()

        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        # decoder_num_classes = pixels per patch = 3 * tubelet * patch^2
        decoder_num_classes = 3 * tubelet_size * patch_size ** 2

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size,
            embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, tubelet_size=tubelet_size,
            num_frames=num_frames)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, tubelet_size=tubelet_size)

        # Project encoder dim → decoder dim
        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

        # Positional embedding for decoder (sinusoidal)
        num_patches = self.encoder.patch_embed.num_patches
        self.pos_embed = get_sinusoid_encoding_table(
            num_patches, decoder_embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'mask_token'}

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, T, H, W] input video
            mask: [B, N] boolean mask (True = masked)
        Returns:
            pred: [B, N_mask, 3*tubelet*patch^2] pixel predictions
        """
        # Encode visible patches
        x_vis = self.encoder(x, mask)             # [B, N_vis, C_enc]
        x_vis = self.encoder_to_decoder(x_vis)    # [B, N_vis, C_dec]

        B, N_vis, C = x_vis.shape

        # Positional embeddings for decoder
        expand_pos = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device)
        pos_vis = expand_pos[~mask].reshape(B, -1, C)     # [B, N_vis, C]
        pos_mask = expand_pos[mask].reshape(B, -1, C)      # [B, N_mask, C]

        # Concat: [visible + mask_tokens] with positional info
        N_mask = pos_mask.shape[1]
        x_full = torch.cat([
            x_vis + pos_vis,
            self.mask_token.expand(B, N_mask, -1) + pos_mask,
        ], dim=1)  # [B, N_vis + N_mask, C_dec]

        # Decode — predict only the masked patches
        pred = self.decoder(x_full, N_mask)  # [B, N_mask, 3*t*p*p]
        return pred


# ─── VisionTransformer for Fine-tuning ────────────────────────────────

class VisionTransformerForFinetune(nn.Module):
    """
    VideoMAE encoder + classification head for action recognition.
    No masking during fine-tuning — all patches are processed.
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=3,
                 num_classes=174, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, fc_drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=0., tubelet_size=2, num_frames=8,
                 use_mean_pooling=True, init_scale=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=num_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # Sinusoidal positional embedding
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values)
            for i in range(depth)
        ])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] input video
        Returns:
            logits: [B, num_classes]
        """
        x = self.patch_embed(x)  # [B, N, C]
        B = x.shape[0]

        pos = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device)
        x = x + pos
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.fc_norm is not None:
            x = self.fc_norm(x.mean(1))  # mean pooling over tokens
        else:
            x = x[:, 0]  # CLS token (not used here)

        x = self.fc_dropout(x)
        x = self.head(x)
        return x

    def load_pretrained(self, checkpoint_path):
        """Load encoder weights from pre-training checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)

        # Map encoder weights to finetune model
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                # Remove 'encoder.' prefix
                new_key = k.replace('encoder.', '')
                # Map patch_embed and blocks
                if new_key.startswith('patch_embed.'):
                    new_state_dict[new_key] = v
                elif new_key.startswith('blocks.'):
                    new_state_dict[new_key] = v
                elif new_key.startswith('norm.'):
                    # Pre-train norm → finetune uses fc_norm for mean pooling
                    pass
                elif new_key == 'pos_embed':
                    # Skip, we use sinusoidal (non-learnable)
                    pass

        # Load matched weights
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"[load_pretrained] Loaded encoder weights. "
              f"Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")
        return msg


# ─── Builder Functions ─────────────────────────────────────────────────

def build_pretrain_model(config):
    """Build PretrainVisionTransformer from config dict."""
    cfg = config['model']
    model = PretrainVisionTransformer(
        img_size=cfg['img_size'],
        patch_size=cfg['patch_size'],
        encoder_embed_dim=cfg['encoder_embed_dim'],
        encoder_depth=cfg['encoder_depth'],
        encoder_num_heads=cfg['encoder_num_heads'],
        decoder_embed_dim=cfg['decoder_embed_dim'],
        decoder_depth=cfg['decoder_depth'],
        decoder_num_heads=cfg['decoder_num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        qkv_bias=cfg['qkv_bias'],
        drop_rate=cfg['drop_rate'],
        attn_drop_rate=cfg['attn_drop_rate'],
        drop_path_rate=cfg['drop_path_rate'],
        tubelet_size=cfg['tubelet_size'],
        num_frames=cfg['num_frames'],
    )
    return model


def build_finetune_model(config):
    """Build VisionTransformerForFinetune from config dict."""
    cfg = config['model']
    model = VisionTransformerForFinetune(
        img_size=cfg['img_size'],
        patch_size=cfg['patch_size'],
        num_classes=cfg['num_classes'],
        embed_dim=cfg['encoder_embed_dim'],
        depth=cfg['encoder_depth'],
        num_heads=cfg['encoder_num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        qkv_bias=cfg['qkv_bias'],
        drop_rate=cfg['drop_rate'],
        attn_drop_rate=cfg['attn_drop_rate'],
        drop_path_rate=cfg['drop_path_rate'],
        tubelet_size=cfg['tubelet_size'],
        num_frames=cfg['num_frames'],
    )
    return model

def build_linear_probe_model(config, pretrain_path=None):
    """
    Build a frozen VisionTransformerForFinetune + a fresh linear head.
    Only model.head is trainable; the entire encoder is frozen.
    """
    # Reuse the same constructor as fine-tuning
    model = build_finetune_model(config)

    # Load pre-trained encoder weights
    import os
    if pretrain_path and os.path.exists(pretrain_path):
        print(f"  Loading pre-trained weights from: {pretrain_path}")
        model.load_pretrained(pretrain_path)
    else:
        print("  WARNING: No pre-trained weights loaded.")

    # Freeze the entire model (encoder + original head)
    for param in model.parameters():
        param.requires_grad = False

    # Replace head with a fresh, trainable linear layer
    # in_features == embed_dim, confirmed from VisionTransformerForFinetune
    model.head = nn.Linear(model.head.in_features, config['model']['num_classes'])
    # model.head is now the only module with requires_grad=True

    return model