import torch
import torch.nn as nn
from typing import Callable, Tuple
from ultralytics.nn.modules import Conv

def tome_with_2d_distance(metric: torch.Tensor, is_upsample: bool) -> Tuple[Callable, Callable]:
    """
    基于ToMe和二维距离的软匹配方法，适配YOLOv8。
    固定参数 k=4, x=3。
    """
    k = 4  # 固定 k 为 4
    x = 3  # 固定 x 为 3

    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        indices = torch.arange(x.shape[1], device=metric.device)
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        a_indices = indices[:a.shape[1] * (k - 1)].view(-1, k)[:, :k - 1].reshape(-1)
        b_indices = indices[(k - 1)::k]
        a_indices = a_indices.unsqueeze(0).expand(x.shape[0], -1)
        b_indices = b_indices.unsqueeze(0).expand(x.shape[0], -1)
        return a, b, a_indices, b_indices

    def get_2d_coordinates(indices):
        side_len = int(indices.shape[1] ** 0.5)  # 计算正方形边长
        y_coords = indices // side_len
        x_coords = indices % side_len
        return torch.stack([y_coords, x_coords], dim=-1)

    def get_nearest_neighbors(a_coords, b_coords, x):
        distances = torch.cdist(a_coords.float(), b_coords.float())
        nearest_indices = distances.topk(x, largest=False).indices
        return nearest_indices

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b, a_idx, b_idx = split(metric)
        r = a.shape[1]
        
        a_coords = get_2d_coordinates(a_idx)
        b_coords = get_2d_coordinates(b_idx)
        
        nearest_indices = get_nearest_neighbors(a_coords, b_coords, x)
        
        batch, na, c = a.shape
        similarity = torch.empty(batch, na, x, device=metric.device)
        for b_idx in range(batch):
            for n1_idx in range(na):
                selected_indices = nearest_indices[b_idx, n1_idx]
                selected_b = b[b_idx, selected_indices, :]
                similarity[b_idx, n1_idx, :] = a[b_idx, n1_idx, :].unsqueeze(0) @ selected_b.transpose(0, 1)
        
        _, dst_idx_pre = similarity.max(dim=-1)
        dst_idx_pre_expanded = dst_idx_pre.unsqueeze(-1)
        dst_idx = torch.gather(nearest_indices, 2, dst_idx_pre_expanded)
        dst_idx.unsqueeze(-1)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst, _, _ = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)
        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge

class XCY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY, self).__init__()
        self.conv = Conv(in_ch, out_ch, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # Flatten and transpose to (B, H*W, C)
        merge, _ = tome_with_2d_distance(x_flat, is_upsample=False)
        x_flat = merge(x_flat)
        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)  # Reshape back to (B, C, H/2, W/2)
        x = self.conv(x)
        return x
