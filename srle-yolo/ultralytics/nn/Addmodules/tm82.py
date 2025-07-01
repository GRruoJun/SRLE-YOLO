import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    """计算卷积所需的填充。"""
    if p is None:
        p = (k - 1) // 2
    return p

class Conv(nn.Module):
    """标准卷积层，带有(输入通道, 输出通道, 卷积核大小, 步长, 填充, 组数, 扩张, 激活函数)参数。"""

    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """初始化卷积层并添加批归一化与激活函数。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """前向传播，应用卷积、批归一化和激活函数。"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """应用激活函数后的卷积操作。"""
        return self.act(self.conv(x))

class BipartiteSoftMatching(nn.Module):
    def __init__(self, r: int, is_upsample: bool):
        super(BipartiteSoftMatching, self).__init__()
        self.r = r
        self.is_upsample = is_upsample

    def forward(self, metric: torch.Tensor):
        protected = 0
        t = metric.shape[1]
        r = min(1008611, (t - protected) // 2)

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            #unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            if self.is_upsample:
                dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)
            else:
                dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            # return torch.cat([unm, dst], dim=1)
            return  dst
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

class LearnableTokenPruning(nn.Module):
    def __init__(self, token_dim, temperature=1.0, lambda_reg=0.01):
        super(LearnableTokenPruning, self).__init__()
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # 可学习的32位阈值
        self.fc = nn.Linear(token_dim, 1)  # 用于生成得分的全连接层

    def forward(self, tokens):
        scores = self.fc(tokens)  # 计算每个 token 的得分
        scores = scores / self.temperature  # 控制 soft mask 的柔软度
        soft_mask = torch.sigmoid(scores - self.threshold)  # 生成 soft mask

        # 应用 soft mask 到 tokens
        pruned_tokens = tokens * soft_mask

        # 对被修剪的部分用0补齐
        pruned_tokens = torch.where(soft_mask > 0.5, pruned_tokens, torch.zeros_like(pruned_tokens))

        # 计算 L1 正则化损失
        l1_loss = self.lambda_reg * soft_mask.abs().mean()

        return pruned_tokens, l1_loss

class AdaptiveFusion(nn.Module):
    def __init__(self, eps=1e-8):
        super(AdaptiveFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)  # 可学习权重
        self.eps = eps  # 避免除零错误

    def forward(self, similarity_scores, spatial_scores):
        weights = nn.ReLU6()(self.weights)  # 使用ReLU6激活函数确保权重在0到6之间
        fuse_weights = weights / (torch.sum(weights) + self.eps)  # 归一化权重使它们的和为1
        return fuse_weights[0] * similarity_scores + fuse_weights[1] * spatial_scores  # 加权融合

class XCY2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY2, self).__init__()
        self.bipartite_down = BipartiteSoftMatching(r=10086110000, is_upsample=False)  # 基于相似度的 bipartite 下采样
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # 1x1卷积层用于通道调整
        self.adaptive_fusion = AdaptiveFusion()  # 自适应融合模块
        self.pruning = None  # LearnableTokenPruning 模块会在第一次调用 forward 时初始化

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        
        # 自动计算 token_dim
        token_dim = C
        
        # 初始化 LearnableTokenPruning 模块
        if self.pruning is None:
            self.pruning = LearnableTokenPruning(token_dim)
        
        # 并行处理路径1：基于相似度打分进行合并
        merge, _ = self.bipartite_down(x_flat)  # 基于相似度得分进行 bipartite 合并
        x_similarity = merge(x_flat)  # 处理后的特征图
        merge, _ = self.bipartite_down(x_similarity)
        x_similarity = merge(x_similarity)  # 处理后的特
        x_similarity  = x_similarity .transpose(1, 2).view(B, C,H // 2, W // 2)
        # 并行处理路径2：先修剪再平均池化
        x_pruned, l1_loss = self.pruning(x_flat)  # 先进行修剪操作
        x_pruned = x_pruned.transpose(1, 2).view(B, C, H, W)  # 恢复特征图形状
        x_pruned = F.avg_pool2d(x_pruned, kernel_size=2, stride=2)  # 应用平均池化进行下采样

        # 自适应融合
        x_fused = self.adaptive_fusion(x_similarity, x_pruned)  # 对两种方式下采样后的特征图进行加权融合

        # 通过 1x1 卷积调整通道数
        x_fused = self.conv(x_fused)

        return x_fused