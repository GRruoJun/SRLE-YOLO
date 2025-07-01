import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoarseBlockAttention(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        S = self.patch_size
        assert H % S == 0 and W % S == 0

        # Step 1: 块平均池化
        x_blocks = x.unfold(2, S, S).unfold(3, S, S)  # (B, C, H//S, W//S, S, S)
        x_blocks = x_blocks.contiguous().view(B, C, -1, S * S)
        x_avg = x_blocks.mean(-1)  # (B, C, N_blocks)
        x_avg = x_avg.permute(0, 2, 1)  # (B, N, C)

        # Step 2: 计算 Q, K
        Q = self.q_proj(x_avg)  # (B, N, C)
        K = self.k_proj(x_avg)  # (B, N, C)

        # Step 3: 计算 A_m + 分布扩散为 A
        A_m = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(C), dim=-1)  # (B, N, N)
        A = A_m.repeat_interleave(S * S, dim=1).repeat_interleave(S * S, dim=2)  # 粗粒度扩散 → (B, hw, hw)

        # Step 4: 计算 V
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, hw, C)
        V = self.v_proj(x_flat)  # (B, hw, C)

        out = torch.bmm(A, V)  # (B, hw, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.gap(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class GLFF(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super().__init__()

        # 每个分支的卷积 + 注意力
        self.conv3x3 = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(3)
        ])
        self.coarse_attn = nn.ModuleList([
            CoarseBlockAttention(in_channels, patch_size) for _ in range(3)
        ])

        # 通道注意力（用于concat后的3c通道）
        self.channel_attn = ChannelAttention(in_channels * 3)

        # 3D卷积 + 通道压缩
        self.conv3d = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, features):
        """
        输入: [F1, F2, F3] 其中
        - F1: (B, C, H/2, W/2)
        - F2: (B, C, H, W)
        - F3: (B, C, 2H, 2W)
        返回:
        - F_out: (B, C, H, W)
        """
        F1, F2, F3 = features
        B, C, H, W = F2.shape

        # 尺度对齐
        F1 = F.interpolate(F1, size=(H, W), mode='bilinear', align_corners=False)
        F3 = F.adaptive_avg_pool2d(F3, output_size=(H, W))

        inputs = [F1, F2, F3]
        processed = []

        for i in range(3):
            conv_feat = self.conv3x3[i](inputs[i])
            attn_feat = self.coarse_attn[i](inputs[i])
            out = conv_feat + attn_feat
            processed.append(out)

        # 拼接后通道注意力
        concat_feat = torch.cat(processed, dim=1)  # (B, 3C, H, W)
        weighted_feat = self.channel_attn(concat_feat)  # (B, 3C, H, W)

        # 拆成3段 → 堆叠 → 3D卷积
        F1_, F2_, F3_ = torch.chunk(weighted_feat, chunks=3, dim=1)
        stack3d = torch.stack([F1_, F2_, F3_], dim=2)  # (B, C, 3, H, W)
        out = self.conv3d(stack3d)[:, :, 1, :, :]  # 中间尺度输出 → (B, C, H, W)
        return self.out_conv(out)
