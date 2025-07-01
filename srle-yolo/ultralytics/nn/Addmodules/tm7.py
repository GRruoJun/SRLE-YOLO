import torch #7改进师兄的代码
import torch.nn as nn
from typing import Callable, Tuple
from ultralytics.nn.modules import Conv

def tome_with_2d_distance(metric: torch.Tensor, is_upsample: bool) -> Tuple[Callable, Callable]:
    """
    基于ToMe和二维距离的软匹配方法，适配YOLOv8。
    固定参数 k=4, x=3。
    """
    k = 4  # 固定 k 为 4
    x = 8  # 固定 x 为 3

    if k <= 1:
        return do_nothing, do_nothing  # 如果 k <= 1，则返回空操作

    def split(x):
        indices = torch.arange(x.shape[1], device=metric.device)  # 生成 x 的索引
        t_rnd = (x.shape[1] // k) * k  # 保证 t_rnd 是 k 的整数倍
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])  # 将 x 切分为 k 个部分
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),  # a 包含前 k-1 个部分
            x[:, :, (k - 1), :],  # b 包含第 k 部分
        )
        a_indices = indices[:a.shape[1] * (k - 1)].view(-1, k)[:, :k - 1].reshape(-1)  # 生成 a 的索引
        b_indices = indices[(k - 1)::k]  # 生成 b 的索引
        a_indices = a_indices.unsqueeze(0).expand(x.shape[0], -1)  # 扩展 a_indices 以匹配批量大小
        b_indices = b_indices.unsqueeze(0).expand(x.shape[0], -1)  # 扩展 b_indices 以匹配批量大小
        return a, b, a_indices, b_indices  # 返回 a, b 及其对应索引

    def get_2d_coordinates(indices):
        side_len = int(indices.shape[1] ** 0.5)  # 根据索引长度计算正方形的边长
        y_coords = indices // side_len  # 计算 y 坐标
        x_coords = indices % side_len  # 计算 x 坐标
        return torch.stack([y_coords, x_coords], dim=-1)  # 将 x, y 坐标堆叠

    def get_nearest_neighbors(a_coords, b_coords, x):
        distances = torch.cdist(a_coords.float(), b_coords.float())  # 计算 a 和 b 坐标的欧氏距离
        nearest_indices = distances.topk(x, largest=False).indices  # 找到距离最近的 x 个邻居
        return nearest_indices  # 返回最近邻索引

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  # 对 metric 进行归一化处理
        a, b, a_idx, b_idx = split(metric)  # 将 metric 分割为 a 和 b
        r = a.shape[1]  # 获取 a 的长度
        
        a_coords = get_2d_coordinates(a_idx)  # 获取 a 的 2D 坐标
        b_coords = get_2d_coordinates(b_idx)  # 获取 b 的 2D 坐标
        
        nearest_indices = get_nearest_neighbors(a_coords, b_coords, x)  # 获取最近邻索引
        
        batch, na, c = a.shape  # 获取 a 的维度
        batch_indices = torch.arange(batch).unsqueeze(1).repeat(1, na)  # shape (batch, na)
        selected_b = b[batch_indices.unsqueeze(-1), nearest_indices]  # shape (batch, na, 3, c)
        similarity = torch.matmul(a.unsqueeze(2), selected_b.transpose(2, 3)).squeeze(2)  # shape (batch, na, 3)
        
        _, dst_idx_pre = similarity.max(dim=-1)  # 在相似度矩阵中找到最大值索引
        dst_idx_pre_expanded = dst_idx_pre.unsqueeze(-1)  # 扩展 dst_idx_pre 的维度
        dst_idx = torch.gather(nearest_indices, 2, dst_idx_pre_expanded)  # 获取目标索引
        dst_idx.unsqueeze(-1)  # 增加一个维度

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst, _, _ = split(x)  # 将输入 x 分割为 src 和 dst
        n, _, c = src.shape  # 获取 src 的维度
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)  # 使用 scatter_reduce 对 dst 进行聚合
        return dst  # 返回聚合后的 dst
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)
        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge,unmerge # 返回合并和还原函数

class XCY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY, self).__init__()
        self.conv = Conv(in_ch, out_ch, 1, 1)  # 初始化一个 1x1 的卷积层

    def forward(self, x):
        B, C, H, W = x.shape  # 获取输入的维度信息
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # 将 x 拉平，并调整维度顺序为 (B, H*W, C)
        merge, _ = tome_with_2d_distance(x_flat, is_upsample=False)  # 调用 tome_with_2d_distance 函数，获取 merge 函数
        x_flat = merge(x_flat)  # 对拉平后的 x 进行合并操作
        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)  # 将 x 的形状还原为 (B, C, H/2, W/2)
        x = self.conv(x)  # 通过卷积层处理 x
        return x  # 返回处理后的输出
