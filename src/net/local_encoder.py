import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from src.utils.ops import furthest_point_sample


def fps_downsample(coor, num_group):
    xyz = coor.transpose(1, 2).contiguous() # b, n, 3
    fps_idx = furthest_point_sample(xyz, num_group)
    new_coor = pointnet2_utils.gather_operation(coor, fps_idx)

    return new_coor.detach()

class Local_Encoder(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        '''
        K has to be 16
        '''
        self.sample_stages = cfg.sample_stages
        self.embed_dim = cfg.embed_dim
        self.k = cfg.knn

        self.knn = KNN(k=cfg.knn, transpose_mode=False)

        self.input_trans = nn.Conv1d(in_channels, 16, 1)

        self.encode_layer = nn.Sequential(
                                    nn.Conv2d(16+3, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(32, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2),
                                    nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2),
                                )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(128+3, 128, kernel_size=1, bias=False),
                    nn.GroupNorm(4, 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv2d(128, 128, kernel_size=1, bias=False),
                    nn.GroupNorm(4, 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Conv2d(128, 128, kernel_size=1, bias=False),
                    nn.GroupNorm(4, 128),
                    nn.LeakyReLU(negative_slope=0.2))
                for i in range(len(self.sample_stages))
            ] 
        )

        self.embed_out = nn.Conv1d(128, self.embed_dim, kernel_size=1, bias=False)

    def get_knn_index(self, coor_q, coor_k=None, k=None):
        coor_k = coor_k if coor_k is not None else coor_q
        # coor: bs, 3, np
        batch_size, _, _ = coor_q.size()
        num_points_k = coor_k.size(2)

        with torch.no_grad():
            if k is None:
                knn_func = self.knn
            else:
                knn_func = KNN(k=k, transpose_mode=False)
            _, idx_raw = knn_func(coor_k, coor_q)  # bs k np
            idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
            idx = idx_raw + idx_base
            idx = idx.view(-1)
            idx_raw = idx_raw.transpose(1, 2).contiguous()  # bs np k
        
        return idx, idx_raw  # [bs*k*np], [bs np k]

    @staticmethod
    def get_graph_feature(coor_q, coor_k, x_k, knn_index, k):
        # coor: bs, 3, np, x: bs, c, np
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = coor_q.size(2)
        num_dims = x_k.size(1)

        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[knn_index, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()  # bs c np k

        coor_k = coor_k.transpose(2, 1).contiguous()
        coor_knn = coor_k.view(batch_size * num_points_k, -1)[knn_index, :]
        coor_knn = coor_knn.view(batch_size, k, num_points_q, 3).permute(0, 3, 2, 1).contiguous()  # bs 3 np k
        coor_q = coor_q.view(batch_size, 3, num_points_q, 1).expand(-1, -1, -1, k)  # bs 3 np k
        feature = torch.cat([coor_knn-coor_q, feature], dim=1)    # bs c+3 np k

        return feature, coor_knn

    def forward(self, coor, embed, trans_type="coord_only"):
        # coor: bs, 3, np; embed: bs, d, np;
        if trans_type == "coord_only":
            fea = self.input_trans(coor)
        else:
            fea = torch.cat([coor, embed], dim=1)
            fea = self.input_trans(fea)

        # stage 0
        knn_index, _ = self.get_knn_index(coor)

        fea, _ = self.get_graph_feature(coor, coor, fea, knn_index, self.k)
        fea = self.encode_layer(fea)
        fea = fea.max(dim=-1, keepdim=False)[0]

        for i in range(len(self.sample_stages)):
            # downsample
            coor_q = fps_downsample(coor, self.sample_stages[i])

            # knn feature
            knn_index, _ = self.get_knn_index(coor_q, coor)
            fea, _ = self.get_graph_feature(coor_q, coor, fea, knn_index, self.k)
            fea = self.layers[i](fea)
            fea = fea.max(dim=-1, keepdim=False)[0]
            coor = coor_q

        out_coor = coor
        out_embed = self.embed_out(fea)

        return out_coor, out_embed

