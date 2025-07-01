import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseGrainedSelfAttention(nn.Module):
    def __init__(self, in_channels, block_size=16):
        super(CoarseGrainedSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.block_size = block_size

        # 用于计算Q, K的卷积层
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 输入：B x C x H x W
        B, C, H, W = x.size()

        # 1. 将输入特征图划分为16x16大小的块
        H_new = H // self.block_size  
        W_new = W // self.block_size
        x_coarse = x.view(B, C, H_new, self.block_size, W_new, self.block_size)  # B x C x H' x 16 x W' x 16
        x_coarse = x_coarse.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, C, H_new * W_new, self.block_size * self.block_size)

        # 2. 计算 Q, K
        Q = self.conv_q(x_coarse)  # B x C x (H' * W') x (block_size * block_size)
        K = self.conv_k(x_coarse)  # B x C x (H' * W') x (block_size * block_size)

        # 3. 计算块之间的注意力权重
        attn_scores = torch.matmul(Q.flatten(2).transpose(1, 2), K.flatten(2))  # B x (H' * W') x (H' * W')
        attn_scores = attn_scores / (self.in_channels ** 0.5)  # 缩放
        attn_weights = torch.softmax(attn_scores, dim=-1)  # B x (H' * W') x (H' * W')

        # 4. 构建块与块之间的注意力权重图
        # 计算每个块的权重，并将其还原到原始尺寸
        attn_weights = attn_weights.view(B, H_new, W_new, H_new, W_new)  # B x H' x W' x H' x W'

        # 5. 还原到原始尺寸：将权重图除以16，以保证每个块的权重均匀分布
        attn_weights = attn_weights / (self.block_size ** 2)  # 除以16

        # 6. 将权重应用到原始特征图
        out = x * attn_weights  # 对原始特征图进行加权

        return out
