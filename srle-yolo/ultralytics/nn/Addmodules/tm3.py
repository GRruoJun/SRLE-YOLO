import torch #效果不好的距离限制tome
import torch.nn as nn
from typing import Callable, Tuple
from ultralytics.nn.modules import Conv

class BipartiteSoftMatchingWithDistance(nn.Module):
    def __init__(self, r: int, is_upsample: bool, distance_threshold: float = None):
        super(BipartiteSoftMatchingWithDistance, self).__init__()
        self.r = r
        self.is_upsample = is_upsample
        self.distance_threshold = distance_threshold

    def compute_distance_threshold(self, metric: torch.Tensor) -> float:
        length = metric.shape[0]
        if length < 2:
            # 默认阈值（如果数据不足）
            return 0.5

        # 选择合适的索引来计算距离
        idx = max(1, length // 16)
        distances = torch.cdist(metric[0:1], metric[idx:idx+1], p=2)

        if distances.numel() == 0:
            # 默认阈值（如果距离计算失败）
            return 0.5

        return distances.item()

    def forward(self, metric: torch.Tensor) -> Tuple[Callable, Callable]:
        protected = 0
        t = metric.shape[1]
        r = min(self.r, (t - protected) // 2)

        with torch.no_grad():
            if self.distance_threshold is None:
                self.distance_threshold = self.compute_distance_threshold(metric)

            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            dist_matrix = torch.cdist(a, b, p=2)
            scores[dist_matrix > self.distance_threshold] = float('-inf')

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            if self.is_upsample:
                dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)
            else:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            return torch.cat([unm, dst], dim=1)

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
            n, _, c = unm.shape
            src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst
            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

            return out

        return merge, unmerge

class XCY(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, distance_threshold: float = None):
        super(XCY, self).__init__()
        self.bipartite_down = BipartiteSoftMatchingWithDistance(r=10086110000, is_upsample=False, distance_threshold=distance_threshold)
        self.bipartite_up = BipartiteSoftMatchingWithDistance(r=10086110000, is_upsample=True, distance_threshold=distance_threshold)
        self.conv = Conv(in_ch, out_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)

        merge, _ = self.bipartite_down(x_flat)
        x_flat = merge(x_flat)
        merge, _ = self.bipartite_down(x_flat)
        x_flat = merge(x_flat)

        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)
        x = self.conv(x)
        return x
