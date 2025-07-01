import torch #版本4，可学习的两个参数
import torch.nn as nn
from ultralytics.nn.modules import Conv
from typing import Callable, Tuple

def tome_with_2d_distance(metric: torch.Tensor, is_upsample: bool, sim_weight: nn.Parameter, dist_weight: nn.Parameter) -> Tuple[Callable, Callable]:
    """
    基于ToMe和二维距离的软匹配方法，考虑相似度得分和空间距离得分的加权和。
    """

    def get_2d_coordinates(indices, width):
        # 根据索引计算二维坐标
        y_coords = indices // width  # 计算y坐标
        x_coords = indices % width  # 计算x坐标
        return torch.stack([y_coords, x_coords], dim=-1)  # 堆叠y和x坐标

    metric = metric / metric.norm(dim=-1, keepdim=True)  # 对metric进行归一化处理
    t = metric.shape[1]  # 获取metric的第二个维度大小（通常为序列长度）
    width = int(t ** 0.5)  # 假设图片为正方形，根据t计算宽度
    a, b = metric[..., :t//2, :], metric[..., t//2:, :]  # 将metric张量一分为二，分别作为a和b
    a_idx = torch.arange(t//2, device=metric.device)  # 生成a的索引
    b_idx = torch.arange(t//2, t, device=metric.device)  # 生成b的索引

    a_coords = get_2d_coordinates(a_idx, width)  # 计算a的二维坐标
    b_coords = get_2d_coordinates(b_idx, width)  # 计算b的二维坐标

    similarity_scores = a @ b.transpose(-1, -2)  # 计算a和b之间的相似度得分
    spatial_distances = torch.cdist(a_coords.float(), b_coords.float())  # 计算a和b之间的空间距离
    spatial_scores = 1.0 / (spatial_distances + 1e-6)  # 距离越小，得分越高，计算空间得分

    combined_scores = sim_weight * similarity_scores + dist_weight * spatial_scores  # 将相似度得分和空间得分加权求和

    _, dst_idx = combined_scores.max(dim=-1)  # 找到得分最高的匹配
    dst_idx = dst_idx.unsqueeze(-1)  # 扩展维度以便后续操作

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        a, b = x[..., :t//2, :], x[..., t//2:, :]  # 将输入张量x一分为二
        n, _, c = a.shape  # 获取a的维度信息
        b = b.scatter_reduce(-2, dst_idx.expand(n, t//2, c), a, reduce=mode)  # 根据dst_idx将a的值合并到b中
        return b  # 返回合并后的张量b

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape  # 获取输入张量x的维度信息
        b = x  # 假设x即为b
        a = b.gather(dim=-2, index=dst_idx.expand(n, t//2, c)).to(x.dtype)  # 从b中收集a的值
        out = torch.cat([a, b], dim=-2)  # 将a和b沿第二个维度拼接
        return out  # 返回拼接后的张量

    return merge, unmerge  # 返回merge和unmerge函数

class XCY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY, self).__init__()
        self.sim_weight = nn.Parameter(torch.tensor(0.5))  # 初始化相似度得分权重
        self.dist_weight = nn.Parameter(torch.tensor(0.5))  # 初始化空间距离得分权重
        self.conv = Conv(in_ch, out_ch, 1, 1)  # 初始化卷积层

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入x的形状
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # 将x展平成二维，并交换维度
        merge, _ = tome_with_2d_distance(x_flat, is_upsample=False, sim_weight=self.sim_weight, dist_weight=self.dist_weight)  # 获取merge函数
        x_flat = merge(x_flat)  # 调用merge函数
        merge, _ = tome_with_2d_distance(x_flat, is_upsample=True, sim_weight=self.sim_weight, dist_weight=self.dist_weight)  # 再次获取merge函数
        x_flat = merge(x_flat)  # 再次调用merge函数
        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)  # 将x_flat恢复为原来的形状，并缩小尺寸
        x = self.conv(x)  # 通过卷积层
        return x  # 返回最终输出
