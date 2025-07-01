import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseStripeConv(nn.Module):
    """一个深度条形卷积: 1xk → kx1"""
    def __init__(self, c, k):
        super().__init__()
        self.dw_conv1 = nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, k // 2), groups=c)
        self.dw_conv2 = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), groups=c)

    def forward(self, x):
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        return x


class MFI_DW(nn.Module):
    def __init__(self, c, k1=3, k2=5, k3=7):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in [k1, k2, k3]:
            self.branches.append(DepthwiseStripeConv(c, k))
        self.identity = nn.Identity()
        self.project = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        out = self.identity(x)
        for branch in self.branches:
            out += branch(x)
        A = self.project(out)
        return A * x  # 通道加权后相乘（逐元素）


class MFI_Bottleneck(nn.Module):
    def __init__(self, c, k1=3, k2=5, k3=7):
        super().__init__()
        self.conv3x3 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.mfi_dw = MFI_DW(c, k1, k2, k3)

    def forward(self, x):
        out = self.conv3x3(x)
        out = self.mfi_dw(out)
        return out + x  # 残差连接


class LWC_C2f(nn.Module):
    def __init__(self, c, n=2, k1=3, k2=5, k3=7):
        super().__init__()
        self.expand = nn.Conv2d(c, 2 * c, kernel_size=1)
        self.blocks = nn.ModuleList([MFI_Bottleneck(c, k1, k2, k3) for _ in range(n)])
        self.fuse = nn.Conv2d((2 + n - 1) * c, c, kernel_size=1)  # Y1, Y2, ..., Yn

    def forward(self, x):
        x = self.expand(x)
        y1, y2 = x.chunk(2, dim=1)  # 通道拆分

        outputs = [y1, y2]
        y = y2
        for block in self.blocks:
            y = block(y)
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        return self.fuse(out)
