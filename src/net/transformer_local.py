import torch
from torch import nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, head_num=8, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()

        assert dim % head_num == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(dim, dim)

        self.n_head = head_num

    def forward(self, x, explicit_atten=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        if explicit_atten is not None:
            att[:, -1, :, :] = explicit_atten

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y

class Block(nn.Module):

    def __init__(self, dim, head_num, attn_pdrop=0.0, resid_pdrop=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalSelfAttention(dim, head_num=head_num, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, explicit_atten=None):
        x = x + self.attn(self.norm1(x), explicit_atten)
        x = x + self.mlp(self.norm2(x))
        return x

def pos_encoding_sin_wave(coor):
    D = 64
    device = coor.device
    # define sin wave freq
    freqs = torch.arange(D, dtype=torch.float).to(device)
    freqs = math.pi * (2**freqs)  # D

    freqs = freqs.view(1, 1, 1, -1)
    coor = coor.unsqueeze(-1) # B, N, 3, 1
    k = coor * freqs # B, N, 3, D
    s = torch.sin(k) # B, N, 3, D
    c = torch.cos(k) # B, N, 3, D
    x = torch.cat([s,c], -1) # B, N, 3, 2D
    x = torch.flatten(x, start_dim=2, end_dim=3) # B, N, 6D
    return x

class Transformer_Local(nn.Module):
    def __init__(self, cfg, input_channel):
        super().__init__()
        mid_embed_dim = input_channel
        num_blocks = cfg.num_blocks
        num_heads = cfg.num_heads
        attn_pdrop = cfg.attn_pdrop
        resid_pdrop = cfg.resid_pdrop
        self.wave_pos_embed = cfg.wave_pos_embed

        if self.wave_pos_embed:
            self.pos_embed = nn.Sequential(
                nn.Linear(64*6, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(128, mid_embed_dim)
            )
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(128, mid_embed_dim)
            )
        self.input_proj = nn.Sequential(
            nn.Linear(input_channel, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, mid_embed_dim)
        )

        self.local_query_embed = nn.Parameter(torch.zeros(mid_embed_dim))
        self.local_query_embed.data.normal_(mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([Block(dim=mid_embed_dim, head_num=num_heads, \
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop) for _ in range(num_blocks)])

        self.ln_f = nn.LayerNorm(mid_embed_dim)

    def forward(self, vis_coor, vis_embed, mis_coor, mis_embed=None):
        # vis_coor: bs, np, 3; vis_embed: bs, np, d;
        # mis_coor: bs, nq, 3; mis_embed: bs, nq, d;
        # output: bs, 3, N; bs C(128) N

        _bs, _np, _d = vis_embed.shape
        if self.wave_pos_embed:
            local_pos_embed = pos_encoding_sin_wave(vis_coor)
            local_pos_embed = self.pos_embed(local_pos_embed)
            global_pos_embed = pos_encoding_sin_wave(mis_coor)
            global_pos_embed = self.pos_embed(global_pos_embed)
        else:
            local_pos_embed =  self.pos_embed(vis_coor)
            global_pos_embed =  self.pos_embed(mis_coor)
        vis_embed = self.input_proj(vis_embed)
        vis_embed = vis_embed + local_pos_embed
        mis_embed = self.local_query_embed[None, None, :] + global_pos_embed

        fea_embed = torch.cat([vis_embed, mis_embed], dim=1)
        for i, blk in enumerate(self.blocks):
            fea_embed = blk(fea_embed)

        return fea_embed

