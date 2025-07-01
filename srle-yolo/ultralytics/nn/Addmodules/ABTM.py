import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from typing import Callable, Tuple

# 定义一个融合模块，用于融合相似度得分和距离得分
class Fusion(nn.Module):
    def __init__(self, eps=1e-8):
        super(Fusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)  # 可学习权重
        self.eps = eps  # 避免除零错误

    def forward(self, similarity_scores, spatial_scores):
        # 使用ReLU6激活函数确保权重在0到6之间
        weights = nn.ReLU6()(self.weights)
        # 归一化权重使它们的和为1
        fuse_weights = weights / (torch.sum(weights) + self.eps)
        # 加权融合相似度得分和空间距离得分
        return fuse_weights[0] * similarity_scores + fuse_weights[1] * spatial_scores

# 修改后的 tome_with_2d_distance 函数，使用 Fusion 模块进行加权融合
def tome_with_2d_distance(metric: torch.Tensor, is_upsample: bool) -> Tuple[Callable, Callable]:
    def get_2d_coordinates(indices, width):
        y_coords = indices // width  # 计算 y 坐标
        x_coords = indices % width  # 计算 x 坐标
        return torch.stack([y_coords, x_coords], dim=-1)  # 合并坐标为二维张量

    # 归一化相似度得分
    metric = metric / metric.norm(dim=-1, keepdim=True)
    t = metric.shape[1]
    a, b = metric[..., :t//2, :], metric[..., t//2:, :]  # 将张量一分为二
    a_idx = torch.arange(t//2, device=metric.device)
    b_idx = torch.arange(t//2, t, device=metric.device)

    # 计算宽高
    width = int(torch.sqrt(torch.tensor(t, dtype=torch.float32)))
    a_coords = get_2d_coordinates(a_idx, width)
    b_coords = get_2d_coordinates(b_idx, width)

    # 计算相似度得分和空间距离得分
    similarity_scores = a @ b.transpose(-1, -2)  # 计算相似度得分
    spatial_distances = torch.cdist(a_coords.float(), b_coords.float())  # 计算空间距离
    spatial_scores = 1.0 / (spatial_distances + 1e-6)  # 距离越小得分越高

    # 初始化 Fusion 模块
    fusion = Fusion()
    # 使用 Fusion 模块对相似度得分和空间距离得分进行加权融合
    combined_scores = fusion(similarity_scores, spatial_scores)

    # 获取得分最高的匹配
    _, dst_idx = combined_scores.max(dim=-1)
    dst_idx = dst_idx.unsqueeze(-1)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, b = x[..., :t//2, :], x[..., t//2:, :]  # 将输入张量一分为二
        n, _, c = a.shape
        b = b.scatter_reduce(-2, dst_idx.expand(n, t//2, c), a, reduce=mode)  # 将张量合并到b中
        return b  # 返回合并后的张量

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        b = x
        a = b.gather(dim=-2, index=dst_idx.expand(n, t//2, c)).to(x.dtype)  # 将b张量中的信息复制回a
        out = torch.cat([a, b], dim=-2)  # 将a和b合并
        return out  # 返回恢复后的张量

    return merge, unmerge

# 定义XCY类，继承自nn.Module
class XCY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY, self).__init__()
        self.conv = Conv(in_ch, out_ch, 1, 1)  # 卷积层

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # 展平图像
        # 使用tome_with_2d_distance进行匹配
        merge, _ = tome_with_2d_distance(x_flat, is_upsample=False)
        x_flat = merge(x_flat)
        merge, _ = tome_with_2d_distance(x_flat, is_upsample=True)
        x_flat = merge(x_flat)
        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)  # 恢复形状
        x = self.conv(x)  # 通过卷积层
        return x  # 返回结果
