import torch
import os
from src.utils.ops import compute_rbf

def m_dist(ref_points, ref_sigmas, ref_rotates, query_points):
    r"""
    input:
    ref_points: (B, ng, 3)
    ref_sigmas: (B, ng, 3)
    ref_rotates: (B, ng, 9)
    query_points: (B, N, 3) 
    
    output:
    dist: (B, N, ng) 
    """
    B, N = query_points.shape[:2]

    # to local coord
    points_local = query_points[:, :, None, :] - ref_points[:, None, :, :]
    points_local = torch.flatten(points_local, start_dim=0, end_dim=2)  # B*N*ng, 3

    ref_sigmas = ref_sigmas.unsqueeze(1).repeat(1, N, 1, 1).contiguous()
    ref_sigmas = torch.flatten(ref_sigmas, start_dim=0, end_dim=2)
    ref_rotates = ref_rotates.unsqueeze(1).repeat(1, N, 1, 1).contiguous()
    ref_rotates = torch.flatten(ref_rotates, start_dim=0, end_dim=2).view(-1, 3, 3)

    points_local = torch.bmm(ref_rotates, points_local.unsqueeze(-1))[:, :, 0]
    dist = torch.exp((-(points_local**2) * ref_sigmas).sum(dim=-1))  # B*N*ng
    dist = dist.view(B, N, -1)

    return dist

def accum_conf_drop_torch(dist_mat, drop_count):
    accum_sum = dist_mat.sum(dim=-1)
    keep_mask = dist_mat.new_ones([dist_mat.shape[0], dist_mat.shape[1]]).bool()
    batch_ids = torch.arange(0, dist_mat.shape[0], device=dist_mat.device)

    for i in range(drop_count):
        _, max_id = accum_sum.max(dim=-1)
        keep_mask[batch_ids, max_id] = 0
        accum_sum[batch_ids, max_id] = -1000.0
        accum_sum = accum_sum - dist_mat[batch_ids, :, max_id]
    return keep_mask

def drop_anchors(anchor_pos, anchor_sigma, anchor_rotate, drop_count, to_drop_feas):
    r"""
    input:
    anchor_pos: (B, nk, 3)  anchor_sigma: (B, nk, 3)  anchor_rotate: (B, nk, 3)  to_drop_feas: (B, nk, d)  

    output: 
    dropped_feas: (B, nc, d)
    """

    with torch.no_grad():
        _B, _nk = anchor_pos.shape[:2]
        keep_count = _nk - drop_count

        dist_mat = compute_rbf(anchor_pos, anchor_pos, anchor_rotate, anchor_sigma)
        # dist_mat = m_dist(anchor_pos, anchor_sigma, anchor_rotate, anchor_pos)  # B, nk, nk
        keep_mask = accum_conf_drop_torch(dist_mat, drop_count).bool() # B, nk

    dropped_feas = to_drop_feas[keep_mask].view(_B, keep_count, -1)

    return dropped_feas

def depth2pc(depth, cam_ex, cam_in, valid):
    # get point in camera coordinate
    device = depth.device
    H, W = depth.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pt_cam = torch.zeros((H, W, 3), dtype=torch.float, device=device)
    pt_cam[:, :, 0] = (gx - cam_in[0][2] * cam_in[2][2]) * depth / cam_in[0][0]
    pt_cam[:, :, 1] = (gy - cam_in[1][2] * cam_in[2][2]) * depth / cam_in[1][1]
    pt_cam[:, :, 2] = depth * cam_in[2][2]
    pt_cam = pt_cam[valid]

    # get point in world coordinate
    pt_world = torch.mm(pt_cam, cam_ex[:3, :3].T) + cam_ex[:3, 3]

    # # get view direction in world coordinate
    # cam_pos = cam_ex[:3, 3]
    # pt_dir = pt_world - cam_pos[None, :]
    # pt_dir = pt_dir / torch.sqrt((pt_dir ** 2).sum(dim=1))[:, None]

    # get point index
    pt_idx = torch.arange(H*W, device=device).view(H, W)
    pt_idx = pt_idx[valid]

    return pt_world, pt_idx


def compute_sdf_grid_vis(decoder, anchor_pos, anchor_code, anchor_sigma, anchor_rotate, input_points, 
                            query_k=1, radius=0.02, voxel_size=1.2, step_size=0.01, max_batch=32**3):
    r"""
    input:
    anchor_pos: (nk, 3), anchor_code: (nk, c), anchor_sigma: (nk, 3), anchor_rotate: (nk, 3), input_points: (nv, 3)

    output: 
    """
    device = anchor_pos.device
    decoder.eval()

    radius_mask = radius - 2 * step_size
    voxel_origin = torch.tensor([-voxel_size/2, -voxel_size/2, -voxel_size/2], device=device)
    voxel_shape = voxel_shape = int(voxel_size // step_size + 1)
    index = torch.arange(voxel_shape, device=device)
    gz, gy, gx = torch.meshgrid(index, index, index, indexing='ij')
    index_grid = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
    input_pos = index_grid * step_size + voxel_origin   # N, 3

    # compute valid region
    input_num = input_pos.shape[0]
    input_valid = torch.zeros([input_num], device = device, dtype=torch.bool)
    input_mask = torch.zeros([input_num], device = device, dtype=torch.bool)
    head = 0
    while head < input_num:
        end = min(head+max_batch, input_num)
        input_dist = ((input_pos[head:end, None, :] - input_points[None, :, :])**2).sum(dim=-1).sqrt()
        min_dist, _ = input_dist.min(dim=1)
        input_valid[head:end] = (min_dist <= radius)
        input_mask[head:end] = (min_dist <= radius_mask)
        head += max_batch

    # predict sdf
    anchor_pos.unsqueeze_(0)
    anchor_code.unsqueeze_(0)
    anchor_sigma.unsqueeze_(0)
    anchor_rotate.unsqueeze_(0)
    input_pos_valid = input_pos[input_valid]
    num_samples = input_pos_valid.shape[0]
    pred_sdf = torch.zeros(num_samples)
    head = 0
    while head < num_samples:
        end = min(head + max_batch, num_samples)
        input_pos_subset = input_pos_valid[head:end].unsqueeze(0).contiguous()

        mdist = compute_rbf(input_pos_subset, anchor_pos, anchor_rotate, anchor_sigma)[0]
        # mdist = m_dist(
        #     anchor_pos, 
        #     anchor_sigma, 
        #     anchor_rotate, 
        #     input_pos_subset)[0] # N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, query_k, dim=-1, largest=True, sorted=True) # N, k

        _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=0, end_dim=1)
        ref_pos = anchor_pos[0][gather_ids].view(1, _N, _k, -1)
        ref_code = anchor_code[0][gather_ids].view(1, _N, _k, -1)

        # to local coord
        points_local = input_pos_subset[:, :, None, :] - ref_pos
        points_local = torch.flatten(points_local, start_dim=0, end_dim=1)
        # prepare anchor code
        ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)
        # prepare knn weight
        mdist_top = mdist_top.view(_N, _k, 1) + 1e-12
        knn_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

        input_data = torch.cat([ref_code, points_local], dim=-1)
        pred = decoder(input_data) 
        pred = (pred * knn_weight).sum(dim=1)[:, 0]
        pred_sdf[head:end] = pred.cpu()

        head += max_batch

    pred_sdf_whole = torch.zeros(voxel_shape ** 3)
    pred_sdf_whole[input_valid] = pred_sdf
    pred_sdf_whole = pred_sdf_whole.view(voxel_shape, voxel_shape, voxel_shape).permute(2,1,0).numpy()
    input_mask = input_mask.view(voxel_shape, voxel_shape, voxel_shape).permute(2,1,0).cpu().numpy()

    return pred_sdf_whole, input_mask, voxel_origin.cpu().numpy(), step_size


def compute_sdf_grid_vis_sep(decoder, anchor_pos, anchor_code, anchor_sigma, anchor_rotate, input_points,
                            radius=0.02, voxel_size=1.2, step_size=0.01, max_batch=32**3):
    r"""
    input:
    anchor_pos: (nk, 3)   anchor_code: (nk, c)    anchor_sigma: (nk, 3)    anchor_rotate: (nk, 3)
    vis_pts: (nv, 3)

    output: 
    """
    device = anchor_pos.device
    decoder.eval()

    radius_mask = radius - 2 * step_size
    voxel_origin = torch.tensor([-voxel_size/2, -voxel_size/2, -voxel_size/2], device=device)
    voxel_shape = voxel_shape = int(voxel_size // step_size + 1)
    index = torch.arange(voxel_shape, device=device)
    gz, gy, gx = torch.meshgrid(index, index, index, indexing='ij')
    index_grid = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
    input_pos = index_grid * step_size + voxel_origin   # N, 3

    # compute valid region
    input_num = input_pos.shape[0]
    anchor_num = anchor_pos.shape[0]
    input_valid = torch.zeros([input_num], device = device, dtype=torch.bool)
    head = 0
    while head < input_num:
        end = min(head+max_batch, input_num)
        input_dist = ((input_pos[head:end, None, :] - input_points[None, :, :])**2).sum(dim=-1).sqrt()
        min_dist, _ = input_dist.min(dim=1)
        input_valid[head:end] = (min_dist <= radius)
        head += max_batch

    # predict sdf
    anchor_pos.unsqueeze_(0)
    anchor_code.unsqueeze_(0)
    anchor_sigma.unsqueeze_(0)
    anchor_rotate.unsqueeze_(0)

    input_pos_valid = input_pos[input_valid]
    num_samples = input_pos_valid.shape[0]
    pred_sdf_list = torch.ones(anchor_num, num_samples)
    mask_list = torch.zeros([anchor_num, num_samples], dtype=torch.bool)
    head = 0
    while head < num_samples:
        end = min(head + max_batch, num_samples)
        input_pos_subset = input_pos_valid[head:end].unsqueeze(0).contiguous()

        mdist = compute_rbf(input_pos_subset, anchor_pos, anchor_rotate, anchor_sigma)[0]
        # mdist = m_dist(
        #     anchor_pos, 
        #     anchor_sigma, 
        #     anchor_rotate, 
        #     input_pos_subset) # 1, N, ng
        mdist_top, mdist_ids_top = mdist.max(dim=-1) # N

        ref_pos = anchor_pos[0, mdist_ids_top].unsqueeze(0)
        ref_code = anchor_code[0, mdist_ids_top].unsqueeze(0)

        # to local coord
        points_local = input_pos_subset - ref_pos
        input_data = torch.cat([ref_code, points_local], dim=-1)
        pred = decoder(input_data)[0, :, 0]

        head_to_end_ids = torch.arange(head, end).long()
        pred_sdf_list[mdist_ids_top, head_to_end_ids] = pred.cpu()
        mask_list[mdist_ids_top, head_to_end_ids] = 1

        head += max_batch
    pred_sdf_whole = torch.ones(anchor_num, voxel_shape ** 3)
    for i in range(anchor_num):
        pred_sdf_whole[i, input_valid] = pred_sdf_list[i]
    pred_sdf_whole = pred_sdf_whole.view(anchor_num, voxel_shape, voxel_shape, voxel_shape).permute(0,3,2,1).numpy()
        
    mask_whole = torch.zeros(anchor_num, voxel_shape ** 3, dtype=torch.bool)
    for i in range(anchor_num):
        mask_whole[i, input_valid] = mask_list[i]
    mask_whole = mask_whole.view(anchor_num, voxel_shape, voxel_shape, voxel_shape).permute(0,3,2,1).numpy()

    for i in range(anchor_num):
        cur_mask = mask_whole[i].copy()
        cur_mask_ori = cur_mask.copy()

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    x_start = max(0, dx)
                    x_end = min(voxel_shape, voxel_shape+dx)
                    y_start = max(0, dy)
                    y_end = min(voxel_shape, voxel_shape+dy)
                    z_start = max(0, dz)
                    z_end = min(voxel_shape, voxel_shape+dz)

                    xx_start = max(0,-dx)
                    xx_end = min(voxel_shape, voxel_shape-dx)
                    yy_start = max(0,-dy)
                    yy_end = min(voxel_shape, voxel_shape-dy)
                    zz_start = max(0,-dz)
                    zz_end = min(voxel_shape, voxel_shape-dz)
                    
                    cur_mask[x_start:x_end, y_start:y_end, z_start:z_end] = \
                        cur_mask[x_start:x_end, y_start:y_end, z_start:z_end] * cur_mask_ori[xx_start:xx_end, yy_start:yy_end, zz_start:zz_end]
        mask_whole[i] = cur_mask

    return pred_sdf_whole, mask_whole, voxel_origin.cpu().numpy(), step_size


def compute_sdf_grid(decoder, anchor_pos, anchor_code, anchor_sigma, anchor_rotate, 
                    voxel_size=1.2, step_size=0.01, max_batch=32**3, query_k=2):
    r"""
    input:
    anchor_pos: (ng, 3)   anchor_code: (ng, c)   anchor_sigma: (ng, 3)   anchor_rotate: (ng, 9)   

    output: 
    """
    device = anchor_pos.device

    voxel_origin = torch.tensor([-voxel_size/2, -voxel_size/2, -voxel_size/2], device=device)
    voxel_shape = int(voxel_size // step_size + 1)
    index = torch.arange(voxel_shape, device=device)
    gz, gy, gx = torch.meshgrid(index, index, index, indexing="ij")
    index_grid = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
    input_pos = index_grid * step_size + voxel_origin   # N, 3

    # predict sdf
    input_num = input_pos.shape[0]
    pred_sdf = torch.zeros(input_num)
    anchor_pos.unsqueeze_(0)
    anchor_code.unsqueeze_(0)
    anchor_sigma.unsqueeze_(0)
    anchor_rotate.unsqueeze_(0)
    input_pos.unsqueeze_(0)
    head = 0
    while head < input_num:
        end = min(head+max_batch, input_num)
        query_points = input_pos[:, head:end, :].contiguous()

        mdist = compute_rbf(query_points, anchor_pos, anchor_rotate, anchor_sigma)[0]
        mdist_top, mdist_ids_top = torch.topk(mdist, query_k, dim=-1, largest=True, sorted=True) # N, k

        _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=0, end_dim=1)
        ref_pos = anchor_pos[0][gather_ids].view(1, _N, _k, -1)
        ref_code = anchor_code[0][gather_ids].view(1, _N, _k, -1)

        # to local coord
        points_local = query_points[:, :, None, :] - ref_pos
        points_local = torch.flatten(points_local, start_dim=0, end_dim=1)
        # prepare anchor code
        ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)
        # prepare knn weight
        mdist_top = mdist_top.view(_N, _k, 1) + 1e-6
        knn_weight = mdist_top / mdist_top.sum(dim=1, keepdim=True)

        input_data = torch.cat([ref_code, points_local], dim=-1)
        pred = decoder(input_data) 
        pred = (pred * knn_weight).sum(dim=1)[:, 0]
        pred_sdf[head:end] = pred.cpu()

        head += max_batch

    pred_sdf = pred_sdf.view(voxel_shape, voxel_shape, voxel_shape).permute(2,1,0).numpy()

    return pred_sdf, voxel_origin.cpu().numpy(), step_size



# def compute_sdf_grid(decoder, anchor_pos, anchor_code, anchor_sigma, anchor_rotate, 
#                     voxel_size=1.2, step_size=0.01, max_batch=64**3, query_k=2):
#     r"""
#     input:
#     anchor_pos: (ng, 3)   anchor_code: (ng, c)   anchor_sigma: (ng, 3)   anchor_rotate: (ng, 9)   

#     output: 
#     """
#     device = anchor_pos.device

#     voxel_origin = torch.tensor([-voxel_size/2, -voxel_size/2, -voxel_size/2], device=device)
#     voxel_shape = int(voxel_size // step_size + 1)
#     index = torch.arange(voxel_shape, device=device)
#     gz, gy, gx = torch.meshgrid(index, index, index, indexing="ij")
#     index_grid = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
#     input_pos = index_grid * step_size + voxel_origin   # N, 3

#     # predict sdf
#     input_num = input_pos.shape[0]
#     pred_sdf = torch.zeros(input_num)
#     anchor_pos.unsqueeze_(0)
#     anchor_code.unsqueeze_(0)
#     anchor_sigma.unsqueeze_(0)
#     anchor_rotate.unsqueeze_(0)
#     input_pos.unsqueeze_(0)

#     valid_query_points_mask = torch.zeros(input_num, dtype=torch.bool)
#     valid_query_ids = []
#     valid_anchor_ids = []
#     valid_query_ids_2 = []
#     valid_anchor_ids_2 = []
#     valid_weights_2 = []
#     valid_anchor_ids_3 = []

#     head = 0
#     while head < input_num:
#         end = min(head+max_batch, input_num)
#         query_points = input_pos[:, head:end, :]

#         mdist = compute_rbf(query_points, anchor_pos, anchor_rotate, anchor_sigma)[0]
#         mdist_top, mdist_ids_top = torch.topk(mdist, query_k, dim=-1, largest=True, sorted=True) # N, k

#         mask_1 = (mdist_top[:, 0] > -1)
#         mask_2 = (mdist_top[:, 1] * 100 > mdist_top[:, 0])
#         valid_query_points_mask[head:end] = mask_1
#         valid_query_ids.append(torch.nonzero(mask_1)[:, 0] + head)
#         valid_anchor_ids.append(mdist_ids_top[mask_1, 0])
#         valid_query_ids_2.append(torch.nonzero(mask_2)[:, 0] + head)
#         valid_anchor_ids_2.append(mdist_ids_top[mask_2, 1])
#         if query_k == 3:
#             valid_anchor_ids_3.append(mdist_ids_top[mask_2, 2])

#         mdist_valid = mdist_top[mask_2]
#         weight_valid = mdist_valid / mdist_valid.sum(dim=1, keepdim=True)
#         valid_weights_2.append(weight_valid.cpu())

#         head += 64**3
#     valid_query_ids = torch.cat(valid_query_ids)
#     valid_anchor_ids = torch.cat(valid_anchor_ids)
#     valid_query_ids_2 = torch.cat(valid_query_ids_2)
#     valid_anchor_ids_2 = torch.cat(valid_anchor_ids_2)
#     valid_weights_2 = torch.cat(valid_weights_2)
#     if query_k == 3:
#         valid_anchor_ids_3 = torch.cat(valid_anchor_ids_3)

#     valid_num_1 = valid_query_ids.shape[0]
#     head = 0
#     while head < valid_num_1:
#         end = min(head+max_batch, valid_num_1)
#         query_points = input_pos[0][valid_query_ids[head:end]]
#         ref_ids = valid_anchor_ids[head:end]

#         ref_pos = anchor_pos[0][ref_ids]
#         ref_code = anchor_code[0][ref_ids]

#         # to local coord
#         points_local = query_points - ref_pos

#         input_data = torch.cat([ref_code, points_local], dim=-1)
#         pred = decoder(input_data)[:, 0].cpu()
#         pred_sdf[valid_query_ids[head:end]] = pred

#         head += max_batch

#     valid_num_2 = valid_query_ids_2.shape[0]
#     head = 0
#     while head < valid_num_2:
#         end = min(head+max_batch, valid_num_2)
#         query_points = input_pos[0][valid_query_ids_2[head:end]]
#         ref_ids = valid_anchor_ids_2[head:end]

#         ref_pos = anchor_pos[0][ref_ids]
#         ref_code = anchor_code[0][ref_ids]

#         # to local coord
#         points_local = query_points - ref_pos

#         input_data = torch.cat([ref_code, points_local], dim=-1)
#         pred = decoder(input_data)[:, 0].cpu()
        
#         pred_sdf[valid_query_ids[head:end]] = \
#             pred_sdf[valid_query_ids[head:end]] * valid_weights_2[head:end, 0] + \
#             pred * valid_weights_2[head:end, 1]

#         if query_k == 3:
#             ref_ids = valid_anchor_ids_3[head:end]

#             ref_pos = anchor_pos[0][ref_ids]
#             ref_code = anchor_code[0][ref_ids]

#             # to local coord
#             points_local = query_points - ref_pos

#             input_data = torch.cat([ref_code, points_local], dim=-1)
#             pred = decoder(input_data)[:, 0].cpu()

#             pred_sdf[valid_query_ids[head:end]] = \
#                 pred_sdf[valid_query_ids[head:end]] + \
#                 pred * valid_weights_2[head:end, 2]

#         head += max_batch

#     pred_sdf = pred_sdf.view(voxel_shape, voxel_shape, voxel_shape).permute(2,1,0).numpy()
#     cur_mask = valid_query_points_mask.view(voxel_shape, voxel_shape, voxel_shape).permute(2,1,0).numpy()

#     cur_mask_ori = cur_mask.copy()
#     for dx in range(-1, 2):
#         for dy in range(-1, 2):
#             for dz in range(-1, 2):
#                 if dx == 0 and dy == 0 and dz == 0:
#                     continue
#                 x_start = max(0, dx)
#                 x_end = min(voxel_shape, voxel_shape+dx)
#                 y_start = max(0, dy)
#                 y_end = min(voxel_shape, voxel_shape+dy)
#                 z_start = max(0, dz)
#                 z_end = min(voxel_shape, voxel_shape+dz)

#                 xx_start = max(0,-dx)
#                 xx_end = min(voxel_shape, voxel_shape-dx)
#                 yy_start = max(0,-dy)
#                 yy_end = min(voxel_shape, voxel_shape-dy)
#                 zz_start = max(0,-dz)
#                 zz_end = min(voxel_shape, voxel_shape-dz)
                
#                 cur_mask[x_start:x_end, y_start:y_end, z_start:z_end] = \
#                     cur_mask[x_start:x_end, y_start:y_end, z_start:z_end] * cur_mask_ori[xx_start:xx_end, yy_start:yy_end, zz_start:zz_end]

#     return pred_sdf, cur_mask, voxel_origin.cpu().numpy(), step_size


def compute_sdf_grid_ldif(decoder, anchor_pos, anchor_code, anchor_sigma, anchor_rotate, 
                    voxel_size=1.2, step_size=0.01, max_batch=64**3, query_k=2):
    r"""
    input:
    anchor_pos: (ng, 3)   anchor_code: (ng, c)   anchor_sigma: (ng, 3)   anchor_rotate: (ng, 9)   

    output: 
    """
    device = anchor_pos.device

    voxel_origin = torch.tensor([-voxel_size/2, -voxel_size/2, -voxel_size/2], device=device)
    voxel_shape = int(voxel_size // step_size + 1)
    index = torch.arange(voxel_shape, device=device)
    gz, gy, gx = torch.meshgrid(index, index, index, indexing="ij")
    index_grid = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
    input_pos = index_grid * step_size + voxel_origin   # N, 3

    # predict sdf
    input_num = input_pos.shape[0]
    pred_sdf = torch.zeros(input_num)
    anchor_pos.unsqueeze_(0)
    anchor_code.unsqueeze_(0)
    anchor_sigma.unsqueeze_(0)
    anchor_rotate.unsqueeze_(0)
    input_pos.unsqueeze_(0)

    head = 0
    while head < input_num:
        end = min(head+64**3, input_num)
        query_points = input_pos[:, head:end, :].contiguous()

        mdist = compute_rbf(query_points, anchor_pos, anchor_rotate, anchor_sigma)[0]
        points_local = query_points[0, :, None, :] - anchor_pos[0, None, :, :]
        ref_code = anchor_code.repeat(query_points.shape[1], 1, 1)

        input_data = torch.cat([ref_code, points_local], dim=-1)

        pred = decoder(input_data)[:, :, 0]
        pred = (mdist * (1 + nonface_pred)).sum(dim=1)
        pred_sdf[head:end] = pred.cpu()

        head += max_batch

    pred_sdf = pred_sdf.view(voxel_shape, voxel_shape, voxel_shape).permute(2,1,0).numpy()
    
    return pred_sdf, voxel_origin.cpu().numpy(), step_size

def compute_sdf_grid_sep(decoder, anchor_pos, anchor_code, anchor_sigma, anchor_rotate, 
                        voxel_size=1.2, step_size=0.01, max_batch=64**3):
    r"""
    input:
    anchor_pos: (ng, 3)   anchor_code: (ng, c)   anchor_sigma: (ng, 3)   anchor_rotate: (ng, 9)   

    output: 
    """
    device = anchor_pos.device

    voxel_origin = torch.tensor([-voxel_size/2, -voxel_size/2, -voxel_size/2], device=device)
    voxel_shape = int(voxel_size // step_size + 1)
    index = torch.arange(voxel_shape, device=device)
    gz, gy, gx = torch.meshgrid(index, index, index, indexing="ij")
    index_grid = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)
    input_pos = index_grid * step_size + voxel_origin   # N, 3

    # compute dist
    input_num = input_pos.shape[0]
    anchor_num = anchor_pos.shape[0]
    pred_sdf_list = torch.ones([anchor_num, input_num]) * 1.0
    mask_list = torch.zeros([anchor_num, input_num], dtype=torch.bool)
    anchor_pos.unsqueeze_(0)
    anchor_code.unsqueeze_(0)
    anchor_sigma.unsqueeze_(0)
    anchor_rotate.unsqueeze_(0)
    input_pos.unsqueeze_(0)
    head = 0

    while head < input_num:
        end = min(head+max_batch, input_num)
        query_points = input_pos[:, head:end, :].contiguous()

        mdist = compute_rbf(query_points, anchor_pos, anchor_rotate, anchor_sigma)
        # mdist = m_dist(anchor_pos, anchor_sigma, anchor_rotate, query_points) # B, N, ng
        mdist_top, mdist_ids_top = torch.topk(mdist, 1, dim=-1, largest=True, sorted=True) # B, N, k

        _B, _N, _k = mdist_top.shape
        gather_ids = torch.flatten(mdist_ids_top, start_dim=1, end_dim=2)
        batch_ids = torch.arange(0, _B, device=gather_ids.device).view(-1, 1).repeat(1, _N*_k).contiguous()
        gather_ids = gather_ids.view(-1)
        batch_ids = batch_ids.view(-1)

        ref_pos = anchor_pos[batch_ids, gather_ids].view(_B, _N, _k, -1)
        ref_code = anchor_code[batch_ids, gather_ids].view(_B, _N, _k, -1)

        # to local coord
        points_local = query_points[:, :, None, :] - ref_pos
        points_local = torch.flatten(points_local, start_dim=0, end_dim=1)
        # prepare anchor code
        ref_code = torch.flatten(ref_code, start_dim=0, end_dim=1)

        input_data = torch.cat([ref_code, points_local], dim=-1)
        pred_0 = decoder(input_data)[:, 0, 0]

        head_to_end_ids = torch.arange(head, end).long()
        pred_sdf_list[mdist_ids_top[0, :, 0], head_to_end_ids] = pred_0.cpu()
        mask_list[mdist_ids_top[0, :, 0], head_to_end_ids] = 1

        head += max_batch
    
    mask_list = mask_list.view(anchor_num, voxel_shape, voxel_shape, voxel_shape).permute(0,3,2,1).numpy()

    for i in range(anchor_num):
        cur_mask = mask_list[i].copy()
        cur_mask_ori = cur_mask.copy()

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    x_start = max(0, dx)
                    x_end = min(voxel_shape, voxel_shape+dx)
                    y_start = max(0, dy)
                    y_end = min(voxel_shape, voxel_shape+dy)
                    z_start = max(0, dz)
                    z_end = min(voxel_shape, voxel_shape+dz)

                    xx_start = max(0,-dx)
                    xx_end = min(voxel_shape, voxel_shape-dx)
                    yy_start = max(0,-dy)
                    yy_end = min(voxel_shape, voxel_shape-dy)
                    zz_start = max(0,-dz)
                    zz_end = min(voxel_shape, voxel_shape-dz)
                    
                    cur_mask[x_start:x_end, y_start:y_end, z_start:z_end] = \
                        cur_mask[x_start:x_end, y_start:y_end, z_start:z_end] * cur_mask_ori[xx_start:xx_end, yy_start:yy_end, zz_start:zz_end]

        mask_list[i] = cur_mask

    pred_sdf_list = pred_sdf_list.view(anchor_num, voxel_shape, voxel_shape, voxel_shape).permute(0,3,2,1).numpy()
    
    return pred_sdf_list, mask_list, voxel_origin.cpu().numpy(), step_size
