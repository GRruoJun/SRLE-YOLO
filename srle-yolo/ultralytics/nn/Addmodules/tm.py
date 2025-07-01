import torch #最初版
import torch.nn as nn
from typing import Callable, Tuple
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
class BipartiteSoftMatching(nn.Module):
    def __init__(self, r: int, is_upsample: bool):
        super(BipartiteSoftMatching, self).__init__()
        self.r = r#匹配的对数
        self.is_upsample = is_upsample#否为上采样模式。如果为False，则为下采样模式

    def forward(self, metric: torch.Tensor) -> Tuple[Callable, Callable]:   #前向传播函数
        #参数:
        ##metric (torch.Tensor): 输入的特征张量。
        
        #返回:
        #merge (Callable): 合并函数。
        #unmerge (Callable): 逆合并函数
        protected = 0
        t = metric.shape[1]#序列长度，metric (torch.Tensor): 输入的特征张量。
        r = min(1008611, (t - protected) // 2)

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)## 归一化metric
            a, b = metric[..., ::2, :], metric[..., 1::2, :]# # 将metric分为两个部分，交替取值
            scores = a @ b.transpose(-1, -2)# 计算匹配得分

            node_max, node_idx = scores.max(dim=-1)# 找到每个节点的最佳匹配
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # 根据得分排序

            unm_idx = edge_idx[..., r:, :]# 未匹配的索引
            src_idx = edge_idx[..., :r, :]# 源索引
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)# 目标索引

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:#合并函数，将源和目标进行合并， 
            #参数:
           #x (torch.Tensor): 输入张量。
            #mode (str): 合并模式，可以是“mean”或“add”。
            src, dst = x[..., ::2, :], x[..., 1::2, :]#分割输入张量
            n, t1, c = src.shape#获取张量形状
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))# 获取未匹配部分
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))#获取匹配部分
            if self.is_upsample:
                dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)# 如果是上采样，目标部分加上源部分
            else:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)# 如果是下采样，根据模式进行合并
            return torch.cat([unm, dst], dim=1) # 合并未匹配部分和目标部分

        def unmerge(x: torch.Tensor) -> torch.Tensor:
            unm_len = unm_idx.shape[1]# 未匹配部分长度
            unm, dst = x[..., :unm_len, :], x[..., unm_len:, :] # 分割输入张量
            n, _, c = unm.shape # 获取张量形状
            src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)) # 获取源部分
            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)# 初始化输出张量

            out[..., 1::2, :] = dst# 交替填充目标部分
            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)# 填充未匹配部分
            out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)# 填充源部分

            return out# 返回还原后的张量

        return merge, unmerge
    
class XCY(nn.Module):
    def __init__(self,in_ch, out_ch,):
        super(XCY, self).__init__()
        self.bipartite_down = BipartiteSoftMatching(r=10086110000, is_upsample=False)
        self.bipartite_up = BipartiteSoftMatching(r=10086110000, is_upsample=True)
        self.conv = Conv(in_ch , out_ch, 1, 1)
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
