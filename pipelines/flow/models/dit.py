import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, freq_emb_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_emb_size, freq_emb_size, bias=True),
            nn.SiLU(),
            nn.Linear(freq_emb_size, freq_emb_size, bias=True)
        )

        half_dim = freq_emb_size // 2
        emb = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor):
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0) # [bs, half_dim]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # [bs, freq_emb_size]
        t_emb = self.mlp(t_freq) # [bs, freq_emb_size]

        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, model_dim: int, n_attn_heads: int, feed_fwd_dim: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim)
        )
        self.multi_head_attn = nn.MultiheadAttention(model_dim, n_attn_heads, batch_first=True)
        self.feed_fwd = nn.Sequential(
            nn.Linear(model_dim, feed_fwd_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(feed_fwd_dim, model_dim)
        )

        # for adaln zero
        nn.init.constant_(self.cond_mlp[-1].weight, 0)
        nn.init.constant_(self.cond_mlp[-1].bias, 0)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: torch.Tensor):
        scale_shift_params = self.cond_mlp(cond).unsqueeze(1)
        scale_1, shift_1, pre_res_1, \
            scale_2, shift_2, pre_res_2 = torch.chunk(scale_shift_params, 6, dim=-1)

        x_norm = self.norm_1(x)
        x_adjusted = x_norm * (1 + scale_1) + shift_1
        attn_out = self.multi_head_attn(
            x_adjusted,
            x_adjusted,
            x_adjusted,
            need_weights=False,
            key_padding_mask=key_padding_mask
        )[0]
        x = x + (pre_res_1 * attn_out)

        norm = self.norm_2(x)
        x_adjusted = norm * (1 + scale_2) + shift_2
        ff_out = self.feed_fwd(x_adjusted)
        x = x + (pre_res_2 * ff_out)

        return x


class TinyDiT(nn.Module):
    def __init__(
            self,
            model_dim: int,
            n_dit_layers: int,
            patch_size: int,
            image_channels: int,
            cond_channels: int,
            image_size: int,
            n_attn_heads: int,
            feed_fwd_dim: int,
            txt_emb_dim: int | None = None,
            max_txt_len: int | None = None,
            n_adaln_cond_cls: int | None = None
        ):
        super().__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.txt_emb_dim = txt_emb_dim
        self.max_txt_len = max_txt_len
        self.patch_conv = nn.Conv2d(
            cond_channels + image_channels,
            model_dim,
            patch_size,
            patch_size
        )
        self.n_patches_side = image_size // patch_size
        self.n_patches = self.n_patches_side ** 2
        self.n_adaln_cond_cls = n_adaln_cond_cls
        self.use_txt_cond = False
        self.use_adaln_cond = False

        self.timestep_emb = TimestepEmbedder(model_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, model_dim) * 0.02)
        if self.txt_emb_dim and self.max_txt_len:
            self.txt_pos_emb = nn.Parameter(torch.randn(1, self.max_txt_len, model_dim) * 0.02)
            # self.txt_proj = nn.Linear(self.txt_emb_dim, model_dim)
            self.txt_mlp = nn.Sequential(
                nn.Linear(self.txt_emb_dim, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            self.use_txt_cond = True
            print("DiT init: using txt embeddings")

        if self.n_adaln_cond_cls:
            self.adaln_cond_emb = nn.Embedding(n_adaln_cond_cls, model_dim)
            self.use_adaln_cond = True
            print("DiT init: using adaLN conditioning")

        self.dit_layers = torch.nn.ModuleList()
        for _ in range(n_dit_layers):
            self.dit_layers.append(
                DiTBlock(model_dim, n_attn_heads, feed_fwd_dim)
            )
        
        self.norm = nn.LayerNorm(model_dim)
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 2)
        )
        self.out_head = nn.Linear(model_dim, (patch_size ** 2) * image_channels)

        nn.init.constant_(self.adaln_mlp[-1].weight, 0)
        nn.init.constant_(self.adaln_mlp[-1].bias, 0)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, txt_cond: torch.Tensor | None = None,
        txt_key_padding_mask: torch.Tensor | None = None, adaln_cond: torch.Tensor | None = None
    ): # TODO: improve cond naming
        bs = x.shape[0]

        # patchify x + c
        x_c = torch.cat([x, cond], dim=1)
        patches = self.patch_conv(x_c) # [bs, dim, n_patches_side, n_patches_side]
        patches = patches.permute(0, 2, 3, 1).reshape(bs, -1, self.model_dim) # [bs, n_patches, dim]
        patches += self.pos_emb # .repeat(bs, 1, 1)
        
        # embed t
        cond_emb = self.timestep_emb(t)

        # adaln cond
        if adaln_cond is not None and self.use_adaln_cond:
            cond_emb += self.adaln_cond_emb(adaln_cond)

        # txt cond
        key_padding_mask = None
        if txt_cond is not None and self.use_txt_cond:
            txt_emb = self.txt_mlp(txt_cond) # [bs, max_txt_len, dim]
            txt_emb += self.txt_pos_emb
            img_key_padding_mask = torch.zeros(bs, patches.shape[1], dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([txt_key_padding_mask, img_key_padding_mask], dim=1)
            patches = torch.cat([txt_emb, patches], dim=1)
        
        # DiT blocks
        for layer in self.dit_layers:
            patches = layer(patches, cond_emb, key_padding_mask)

        # layer norm
        norm_out = self.norm(patches)
        scale_shift_params = self.adaln_mlp(cond_emb).unsqueeze(1)
        scale, shift = torch.chunk(scale_shift_params, 2, dim=-1)
        x = norm_out * (1 + scale) + shift

        if txt_cond is not None and self.use_txt_cond:
            x = x[:, self.max_txt_len:, :]

        # linear and reshape
        out = self.out_head(x) # [bs, n_patches, p*p*c]
        out = out.reshape(
            bs, self.n_patches_side, self.n_patches_side,
            self.image_channels, self.patch_size, self.patch_size
        ).permute(0, 3, 1, 4, 2, 5)
        velocity = out.reshape(bs, self.image_channels, self.image_size, self.image_size)

        return velocity
