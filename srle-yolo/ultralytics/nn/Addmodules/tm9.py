import torch#9双打分
import torch.nn as nn
from ultralytics.nn.modules import Conv
from typing import Callable, Tuple

# 定义一个融合模块，用于融合相似度得分和距离得分
class AdaptiveFusion(nn.Module):
    def __init__(self, eps=1e-8):
        super(AdaptiveFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)  # 可学习权重
        self.eps = eps  # 避免除零错误

    def forward(self, similarity_scores, spatial_scores):
        weights = nn.ReLU6()(self.weights)  # 使用ReLU6激活函数确保权重在0到6之间
        fuse_weights = weights / (torch.sum(weights) + self.eps)  # 归一化权重使它们的和为1
        return fuse_weights[0] * similarity_scores + fuse_weights[1] * spatial_scores  # 加权融合相似度得分和空间距离得分

# 计算二维坐标函数
def get_2d_coordinates(indices, width):
    y_coords = indices // width  # 计算 y 坐标
    x_coords = indices % width  # 计算 x 坐标
    return torch.stack([y_coords, x_coords], dim=-1)  # 合并坐标为二维张量

# 修改后的 tome_with_2d_distance 函数，返回两个独立的合并函数
def tome_with_2d_distance(metric: torch.Tensor) -> Tuple[Callable, Callable]:
    metric = metric / metric.norm(dim=-1, keepdim=True)  # 归一化相似度得分
    t = metric.shape[1]
    
    # 每四个点取一个作为a组，剩余的作为b组
    a_idx = torch.arange(0, t, 4, device=metric.device)  # 选取 a 组索引
    b_idx = torch.tensor([i for i in range(t) if i % 4 != 0], device=metric.device)  # 选取 b 组索引

    a = metric[..., a_idx, :]  # 获取 a 组张量
    b = metric[..., b_idx, :]  # 获取 b 组张量

    # 计算宽高（假设图片是正方形）
    width = int(torch.sqrt(torch.tensor(t, dtype=torch.float32)))
    a_coords = get_2d_coordinates(a_idx, width)
    b_coords = get_2d_coordinates(b_idx, width)

    # 计算相似度得分和空间距离得分
    similarity_scores = a @ b.transpose(-1, -2)  # 计算相似度得分
    spatial_distances = torch.cdist(a_coords.float(), b_coords.float())  # 计算空间距离
    spatial_scores = 1.0 / (spatial_distances + 1e-6)  # 距离越小得分越高

    # 获取得分最高的匹配
    _, sim_dst_idx = similarity_scores.max(dim=-1)
    _, spa_dst_idx = spatial_scores.max(dim=-1)

    sim_dst_idx = sim_dst_idx.unsqueeze(-1)
    spa_dst_idx = spa_dst_idx.unsqueeze(-1)

    # 定义相似度合并函数
    def merge_similarity(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, b = x[..., a_idx, :], x[..., b_idx, :]  # 获取 a 和 b 组
        n, t_a, c = a.shape  # 这里 t_a 是 a 组的数量
        _, t_b, _ = b.shape  # t_b 是 b 组的数量
        
        # 根据 sim_dst_idx 从 b 组中获取对应的元素
        selected_b = torch.gather(b, 1, sim_dst_idx.expand(n, t_a, c))

        # 合并 b 组的元素到 a 组中
        a_new = a + selected_b  # 合并操作

        if mode == "mean":
            a_new = a_new / 2  # 如果是平均模式，则进行平均操作

        return a_new  # 返回合并后的张量

    # 定义空间距离合并函数
    def merge_spatial(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, b = x[..., a_idx, :], x[..., b_idx, :]  # 获取 a 和 b 组
        n, t_a, c = a.shape  # 这里 t_a 是 a 组的数量
        _, t_b, _ = b.shape  # t_b 是 b 组的数量
        
        # 根据 spa_dst_idx 从 b 组中获取对应的元素
        selected_b = torch.gather(b, 1, spa_dst_idx.expand(n, t_a, c))

        # 合并 b 组的元素到 a 组中
        a_new = a + selected_b  # 合并操作

        if mode == "mean":
            a_new = a_new / 2  # 如果是平均模式，则进行平均操作

        return a_new  # 返回合并后的张量

    return merge_similarity, merge_spatial

# 定义XCY类，继承自nn.Module
class XCY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY, self).__init__()
        self.conv = Conv(in_ch, out_ch, 1, 1)  # 卷积层
        self.fusion = AdaptiveFusion()  # 自适应融合模块

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # 展平图像
        
        # 使用tome_with_2d_distance进行匹配
        merge_sim, merge_spatial = tome_with_2d_distance(x_flat)
        
        # 获取相似度和空间距离打分后的特征
        x_flat_sim = merge_sim(x_flat)
        x_flat_spatial = merge_spatial(x_flat)
        
        # 自适应融合
        x_flat_fused = self.fusion(x_flat_sim, x_flat_spatial)
        
        # 重新调整形状
        x = x_flat_fused.transpose(1, 2).view(B, C, H // 2, W // 2)  # 恢复形状
        x = self.conv(x)  # 通过卷积层
        return x  # 返回结果
