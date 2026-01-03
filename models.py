# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import timm
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp, VisionTransformer, Block


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        x2_fuse_every=0,           # >0 to fuse x2 into x every N blocks (early/throughout)
        x2_condition_with_c=False,  # True to FiLM-condition x2 with t+y before its ViT block
        x2_final_fuse=True,        # keep the final add of x2 after all blocks
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x2_embedder = PatchEmbed(input_size, patch_size // 2, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # Learnable CLS tokens for x2 (one per group of 4 patches)
        self.x2_cls_tokens = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # ViT block for processing x2
        # Will be initialized with target dimensions, but may be replaced with pretrained block if dimensions differ
        self.x2_vit_block = Block(
            dim=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True
        )
        # Optional FiLM to condition x2 on c (t+y)
        self.x2_condition_with_c = x2_condition_with_c
        if self.x2_condition_with_c:
            self.x2_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        else:
            self.x2_film = None
        # Projection layers for adapting dimensions if needed (will be created if dimensions don't match)
        self.x2_vit_proj_in = None
        self.x2_vit_proj_out = None
        self.x2_fuse_every = x2_fuse_every
        self.x2_final_fuse = x2_final_fuse
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # Load pre-trained timm ViT weights for x2_vit_block
        self.load_pretrained_vit_weights()

    def load_pretrained_vit_weights(self, vit_model_name='vit_large_patch16_224'):
        """
        Load pre-trained timm ViT weights for x2_vit_block.
        If dimensions don't match, uses projection layers to adapt.
        """
        try:
            # Load pre-trained ViT model
            print(f"[DiT] Loading pre-trained ViT model: {vit_model_name}", flush=True)
            pretrained_vit = timm.create_model(vit_model_name, pretrained=True)
            pretrained_vit.eval()
            
            # Extract the last block
            pretrained_block = pretrained_vit.blocks[-1]
            pretrained_dim = pretrained_vit.embed_dim
            # Get num_heads from the attention module or infer from qkv weight shape
            pretrained_num_heads = None
            # Try multiple ways to get num_heads
            if hasattr(pretrained_block.attn, 'num_heads'):
                pretrained_num_heads = pretrained_block.attn.num_heads
            elif hasattr(pretrained_vit, 'num_heads'):
                pretrained_num_heads = pretrained_vit.num_heads
            else:
                # Infer from qkv weight shape
                # qkv weight shape is (3 * num_heads * head_dim, embed_dim)
                # where embed_dim = num_heads * head_dim
                # So: qkv_out_dim = 3 * embed_dim, and head_dim = embed_dim / num_heads
                # Therefore: qkv_out_dim = 3 * num_heads * (embed_dim / num_heads) = 3 * embed_dim
                # This means we can't directly get num_heads from qkv_out_dim alone
                # But we can calculate: num_heads = embed_dim / head_dim
                # And head_dim = qkv_out_dim / (3 * num_heads) = embed_dim / num_heads
                # So: qkv_out_dim = 3 * embed_dim (always true for standard ViT)
                # We need to infer head_dim. Standard head_dim values: 64 (most common)
                qkv_out_dim = pretrained_block.attn.qkv.weight.shape[0]
                # Calculate head_dim from qkv: head_dim = qkv_out_dim / (3 * num_heads)
                # Since we don't know num_heads, let's use common defaults
                # Common ViT configs: embed_dim 768 -> 12 heads (head_dim=64), 1024 -> 16 heads (head_dim=64)
                if pretrained_dim == 768:
                    pretrained_num_heads = 12
                elif pretrained_dim == 1024:
                    pretrained_num_heads = 16
                elif pretrained_dim == 1280:
                    pretrained_num_heads = 16
                else:
                    # Calculate: assume head_dim = 64 (most common)
                    pretrained_num_heads = pretrained_dim // 64
                    if pretrained_num_heads <= 0 or pretrained_num_heads > 32:
                        pretrained_num_heads = 16  # fallback to common value
                print(f"[DiT] Inferred num_heads={pretrained_num_heads} from embed_dim={pretrained_dim}", flush=True)
            
            print(f"[DiT] Pre-trained ViT block: dim={pretrained_dim}, num_heads={pretrained_num_heads}", flush=True)
            print(f"[DiT] Target x2_vit_block: dim={self.x2_vit_block.norm1.normalized_shape[0]}, num_heads={self.num_heads}", flush=True)
            
            # Check if dimensions match
            if pretrained_dim == self.hidden_size and pretrained_num_heads == self.num_heads:
                # Direct weight loading if dimensions match
                print(f"[DiT] Dimensions match! Loading weights directly...", flush=True)
                self.x2_vit_block.load_state_dict(pretrained_block.state_dict(), strict=True)
                print(f"[DiT] ✓ Successfully loaded pre-trained ViT weights for x2_vit_block", flush=True)
            else:
                # Dimensions don't match - replace block with pretrained dimensions and use projection layers
                print(f"[DiT] Dimensions don't match. Creating block with pretrained dimensions and projection layers...", flush=True)
                
                # Create a new block with pretrained dimensions
                pretrained_mlp_ratio = pretrained_vit.mlp_ratio if hasattr(pretrained_vit, 'mlp_ratio') else 4.0
                self.x2_vit_block = Block(
                    dim=pretrained_dim,
                    num_heads=pretrained_num_heads,
                    mlp_ratio=pretrained_mlp_ratio,
                    qkv_bias=True
                )
                
                # Load pre-trained weights into the new block
                self.x2_vit_block.load_state_dict(pretrained_block.state_dict(), strict=True)
                
                # Create projection layers
                self.x2_vit_proj_in = nn.Linear(self.hidden_size, pretrained_dim)
                self.x2_vit_proj_out = nn.Linear(pretrained_dim, self.hidden_size)
                # Initialize projection layers
                nn.init.xavier_uniform_(self.x2_vit_proj_in.weight)
                nn.init.constant_(self.x2_vit_proj_in.bias, 0)
                nn.init.xavier_uniform_(self.x2_vit_proj_out.weight)
                nn.init.constant_(self.x2_vit_proj_out.bias, 0)
                
                print(f"[DiT] ✓ Successfully loaded pre-trained ViT weights into block with dim={pretrained_dim}", flush=True)
                print(f"[DiT] Using projection layers to adapt dimensions: {self.hidden_size} -> {pretrained_dim} -> {self.hidden_size}", flush=True)
                
        except Exception as e:
            print(f"[DiT] Warning: Could not load pre-trained ViT weights: {e}", flush=True)
            print(f"[DiT] Using randomly initialized weights for x2_vit_block", flush=True)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        # skip = self.x_embedder(x)                                 # preserve pre-block representation
        x2 = self.x2_embedder(x)  # (N, 4T, D)
        
        # Insert CLS tokens after every 4 rows: (N, 4T, D) -> (N, 5T, D)
        N, seq_len, D = x2.shape
        T = seq_len // 4  # Number of groups
        # Reshape to group patches in sets of 4: (N, 4T, D) -> (N, T, 4, D)
        x2 = x2.reshape(N, T, 4, D)
        # Expand learnable CLS tokens: (1, T, D) -> (N, T, 1, D)
        # Use repeat instead of expand to ensure gradients flow properly to the learnable parameter
        x2_cls_tokens_expanded = self.x2_cls_tokens.repeat(N, 1, 1).unsqueeze(2)  # (N, T, 1, D)
        # Concatenate CLS tokens after each group of 4: (N, T, 4, D) + (N, T, 1, D) -> (N, T, 5, D)
        x2 = torch.cat([x2, x2_cls_tokens_expanded], dim=2)  # (N, T, 5, D)
        # Reshape back to sequence: (N, T, 5, D) -> (N, 5T, D)
        x2 = x2.reshape(N, 5 * T, D)
        
        # print(f"[DiT Forward] skip: {skip.shape}", flush=True)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        # Pool x2 from shape (N, 5T, D) to (N, T, D) to match x
        # Need to transpose for avg_pool1d which expects (N, C, L) format
        # x2 = x2.transpose(1, 2)  # (N, D, 5T)
        # x2 = torch.avg_pool1d(x2, kernel_size=5, stride=5) # my approach
        # x2 = -torch.max_pool1d(-x2, kernel_size=5, stride=5) # min pooling (rezghi approach)
        # Extract CLS tokens: take every 5th row starting from index 4 (before transpose, CLS tokens were at positions 4, 9, 14, ...)
        # After transpose and pooling, we need to extract from the original x2 before pooling
        # So we extract from x2 before the pooling operation
        # x2_cls = x2[:, :, 4::5]  # Extract CLS tokens: (N, D, T)
        # x2_cls = x2_cls.transpose(1, 2)  # (N, T, D)
        # x2 = x2.transpose(1, 2)  # (N, T, D)
        # Inject positional info so x2 tokens carry spatial cues like x
        # x2 = x2 + self.pos_embed # turn off for now because it increases FID
        # Optionally condition x2 on c via FiLM before the ViT block
        if self.x2_film is not None:
            shift, scale = self.x2_film(c).chunk(2, dim=1)
            x2 = modulate(x2, shift, scale)
        # Apply ViT block to x2 to ensure same processing as x
        # Use projection layers if dimensions don't match
        if self.x2_vit_proj_in is not None:
            x2 = self.x2_vit_proj_in(x2)  # Project to pre-trained ViT dimension
        x2 = self.x2_vit_block(x2)  # (N, T, D) - processed by pre-trained ViT block
        if self.x2_vit_proj_out is not None:
            x2 = self.x2_vit_proj_out(x2)  # Project back to hidden_size
        # extract CLS tokens from x2
        x2 = x2[:, 4::5, :]  # Extract CLS tokens: (N, T, D)
        # shape of x2 and x
        print(f"[DiT Forward] x2 shape: {x2.shape}, x shape: {x.shape}", flush=True)
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if self.x2_fuse_every and (i + 1) % self.x2_fuse_every == 0:
                x = x + x2
        if self.x2_final_fuse:
            x = x + x2
        
        # print(f"[DiT Forward] ✓ Skip org connection applied across {len(self.blocks)} blocks", flush=True)
        
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        print(f"\n{'='*60}")
        print(f"[CFG] Input: {x.shape}, CFG scale: {cfg_scale}")
        
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        model_out = self.forward(combined, t, y)
        
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        print(f"[CFG] ✓ Guidance applied, output: {torch.cat([eps, rest], dim=1).shape}")
        print(f"{'='*60}\n")
        
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    print("Creating DiT-XL/2 model...")
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    print("Creating DiT-XL/4 model...")
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
