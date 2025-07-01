import torch #备份8
import torch.nn as nn
import torch.nn.functional as F
# from ultralytics.nn.modules import Conv  # 确保 Conv 是一个卷积层
def autopad(k, p=None, d=1):
    """Calculate the padding needed for convolution."""
    if p is None:
        p = (k - 1) // 2
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
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


class XCY1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(XCY1, self).__init__()
        self.bipartite_down = BipartiteSoftMatching(r=10086110000, is_upsample=False)
        self.bipartite_up = BipartiteSoftMatching(r=10086110000, is_upsample=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # 使用 PyTorch 的 Conv2d 代替 Conv
        self.pruning = None

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        
        # 自动计算 token_dim
        token_dim = C
        
        # 初始化 LearnableTokenPruning 模块
        if self.pruning is None:
            self.pruning = LearnableTokenPruning(token_dim)
        # 应用 Learnable Token Pruning 模块
        x_flat, l1_loss = self.pruning(x_flat)
        # Down-sampling
        merge, _ = self.bipartite_down(x_flat)
        x_flat = merge(x_flat)
        # 再次 Down-sampling
        merge, _ = self.bipartite_down(x_flat)
        x_flat = merge(x_flat)
        
        x = x_flat.transpose(1, 2).view(B, C, H // 2, W // 2)
        x = self.conv(x)
        
        return x # 返回输出和正则化损失
