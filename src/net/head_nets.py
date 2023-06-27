import torch
from torch import nn


class ShapeCodeHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embed_pred = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, output_dim)
        )
    def forward(self, input_embed):
        return self.embed_pred(input_embed)


class OffsetHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.offset_pred = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, 3)
        )

    def forward(self, input_embed):
        return self.offset_pred(input_embed)

class SigmaHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.offset_pred = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, 3)
        )

    def forward(self, input_embed):
        return torch.exp(self.offset_pred(input_embed))

class RotateHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rotate_pred = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim, 6)
        )

    def forward(self, input_embed):
        vectors = self.rotate_pred(input_embed)
        v1_raw = vectors[:, :, :3]
        v2_raw = vectors[:, :, 3:]
        v1 = nn.functional.normalize(v1_raw, dim=-1)
        v3 = torch.cross(v1, v2_raw, dim=-1)
        v3 = nn.functional.normalize(v3, dim=-1)
        v2 = torch.cross(v3, v1, dim=-1)

        rotate_matrix = torch.cat([v1, v2, v3], dim=-1)

        return rotate_matrix

class GlobalPointsHead(nn.Module):
    def __init__(self, input_dim, point_num):
        super().__init__()
        self.point_num = point_num

        self.global_pos_pred = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LayerNorm(input_dim*2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim*2, input_dim*2),
            nn.LayerNorm(input_dim*2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(input_dim*2, 3 * self.point_num)
        )


    def forward(self, input_embed):
        global_points = self.global_pos_pred(input_embed).view(-1, self.point_num, 3)    # bs, nq, 3

        return global_points
