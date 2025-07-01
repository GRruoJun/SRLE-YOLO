import torch #可学习的阈值
import torch.nn as nn
from typing import Callable, Tuple
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class BipartiteSoftMatching(nn.Module):
    def __init__(self, r: int, is_upsample: bool):
        super(BipartiteSoftMatching, self).__init__()
        self.r = r  # 匹配的对数
        self.is_upsample = is_upsample  # 是否为上采样模式。如果为False，则为下采样模式
        self.threshold = nn.Parameter(torch.tensor(50, dtype=torch.float32))  # 可学习的阈值参数
      # 注册钩子函数到阈值参数
        self.threshold.register_hook(self.print_threshold)

    def print_threshold(self, grad):
        print(f"Current threshold: {self.threshold.item()}")

    def forward(self, metric: torch.Tensor) -> Tuple[Callable, Callable]:
        
        protected = 0
        t = metric.shape[1]  # 序列长度，metric (torch.Tensor): 输入的特征张量。
        r = min(self.r, (t - protected) // 2)

        metric = metric / metric.norm(dim=-1, keepdim=True)  # 归一化metric
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # 将metric分为两个部分，交替取值
        scores = a @ b.transpose(-1, -2)  # 计算匹配得分

        node_max, node_idx = scores.max(dim=-1)  # 找到每个节点的最佳匹配
        edge_idx = node_max.argsort(dim=-1, descending=True).unsqueeze(-1)  # 根据得分排序

        # 根据阈值过滤
        score_mask = node_max > self.threshold  # 阈值比较
        filtered_edge_idx = edge_idx.masked_select(score_mask.unsqueeze(-1)).view(-1, 1)  # 筛选出大于阈值的索引

        # 如果没有任何边通过阈值过滤，确保我们至少有一个索引
        if filtered_edge_idx.numel() == 0:
            filtered_edge_idx = edge_idx[..., :1, :]  # 使用一个默认索引

        unm_idx = edge_idx[..., r:, :]  # 未匹配的索引
        src_idx = filtered_edge_idx[:r].expand(metric.size(0), -1, 1)  # 源索引
        dst_idx = node_idx.gather(dim=-1, index=src_idx.squeeze(-1))  # 目标索引

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]  # 分割输入张量
            n, t1, c = src.shape  # 获取张量形状

            # 使用mask过滤未匹配部分
            if unm_idx.numel() > 0:
                unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))  # 获取未匹配部分
            else:
                unm = src[..., :0, :]  # 如果没有未匹配部分，创建一个空张量

            src = src.gather(dim=-2, index=src_idx.expand(n, src_idx.shape[1], c))  # 获取匹配部分

            if self.is_upsample:
                dst = dst.scatter_add(-2, dst_idx.unsqueeze(-1).expand(n, src_idx.shape[1], c), src)  # 如果是上采样，目标部分加上源部分
            else:
                dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(-1).expand(n, src_idx.shape[1], c), src, reduce=mode)  # 如果是下采样，根据模式进行合并

            # 检查合并后长度是否小于一半
            combined_length = dst.shape[-2] + unm.shape[-2]
            if combined_length < t1 // 2:
                # 插值填充未匹配部分
                dst = torch.cat([unm, dst], dim=1)  # 合并未匹配部分和目标部分
                dst = F.interpolate(dst, size=(t1,), mode='linear', align_corners=False)  # 插值返回
            else:
                dst = torch.cat([unm, dst], dim=1)  # 直接合并

            return dst

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]  # 未匹配部分长度
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]  # 分割输入张量
            n, _, c = unm.shape  # 获取张量形状
            src = dst.gather(dim=-2, index=dst_idx.unsqueeze(-1).expand(n, src_idx.shape[1], c))  # 获取源部分
            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)  # 初始化输出张量

            out[..., 1::2, :] = dst  # 交替填充目标部分
            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)  # 填充未匹配部分
            out.scatter_(dim=-2, index=(2 * src_idx).expand(n, src_idx.shape[1], c), src=src)  # 填充源部分

            return out  # 返回还原后的张量

        return merge, unmerge

class XCY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY, self).__init__()
        self.bipartite_down = BipartiteSoftMatching(r=10086111000, is_upsample=False)  # 更改r为实际需要的值
        self.bipartite_up = BipartiteSoftMatching(r=10, is_upsample=True)  # 更改r为实际需要的值
        self.conv = Conv(in_ch, out_ch, 1, 1)

    def forward(self, x):
        # Example of down-sampling
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # Flatten and transpose to (B, H*W, C)
        merge, _ = self.bipartite_down(x_flat)
        x_flat = merge(x_flat)
        merge, _ = self.bipartite_down(x_flat)
        x_flat = merge(x_flat)
        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)  # Reshape back to (B, C, H/2, W/2)
        x = self.conv(x)
        return x
