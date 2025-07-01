import torch  # 9,重叠分块 mlp 聚合特征
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['Zoom_cat', 'ScalSeq', 'Add', 'channel_att', 'attention_model']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OverlapBlockAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1, block_size=16, overlap=4):
        super(OverlapBlockAttention, self).__init__()
        # 初始化输入通道数、注意力头数、分块大小和重叠大小
        self.num_heads = num_heads  # 注意力头的数量
        self.head_dim = in_channels // num_heads  # 每个注意力头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子，用于防止注意力值过大
        self.block_size = block_size  # 分块大小，默认是16x16
        self.overlap = overlap  # 重叠区域的大小，默认为4

        # 定义查询、键和值的卷积层，保持通道数不变
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 获取输入特征图的尺寸信息：批次大小、通道数、高度、宽度
        batch_size, channels, height, width = x.size()

        # 计算分块的步幅，步幅等于分块大小减去重叠部分
        stride = self.block_size - self.overlap

        # 存储所有分块的列表
        blocks = []

        # 遍历特征图，按重叠分块的方式提取块
        for i in range(0, height - self.block_size + 1, stride):
            for j in range(0, width - self.block_size + 1, stride):
                # 提取大小为 block_size x block_size 的重叠块
                block = x[:, :, i:i + self.block_size, j:j + self.block_size]
                # 将分块添加到列表中
                blocks.append(block)
        if len(blocks) == 0:
            return x
        # 将所有的分块在第一个维度（批次维度）上进行拼接
        blocks = torch.cat(blocks, dim=0)  # [N*B, C, B, B]

        # # 根据分块的数量更新 batch_size
        # batch_size = blocks.size(0) // (self.block_size * self.block_size)

        # 计算分块的数量和 batch_size，确保 batch_size 正确
        n_blocks = len(blocks) // batch_size
        batch_size = blocks.size(0) // n_blocks  # 修正 batch_size 的计算
        # 对分块计算查询、键和值，转换为适合多头注意力的形式
        query = self.query_conv(blocks).view(batch_size, self.num_heads, self.head_dim, -1).transpose(2, 3)
        key = self.key_conv(blocks).view(batch_size, self.num_heads, self.head_dim, -1).transpose(2, 3)
        value = self.value_conv(blocks).view(batch_size, self.num_heads, self.head_dim, -1).transpose(2, 3)

        # 计算注意力得分，查询与键的转置相乘，并乘以缩放因子
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # 对注意力得分进行 softmax 归一化
        attention = F.softmax(attention, dim=-1)

        # 用注意力矩阵对值进行加权求和，获得输出
        out = torch.matmul(attention, value).transpose(2, 3).contiguous().view(batch_size, channels, self.block_size, self.block_size)

        # 通过一个卷积层输出，并进行残差连接
        out = self.out_conv(out)

        # 将残差连接后的输出返回
        return out + x  # 残差连接

class ScalSeq(nn.Module):
    def __init__(self, inc, channel,block_size=64, overlap=4, num_heads=1):
        super(ScalSeq, self).__init__()
        self.channel = channel  # 将 channel 存储为类的属性
        # 定义不同层的卷积，统一输出通道数
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.block_size = block_size  # 添加 block_size 作为成员变量
        self.overlap = overlap  # 添加 overlap 作为成员变量
        self.num_heads = num_heads  # 添加 num_heads 作为成员变量

        # 定义注意力模块，用于不同尺度的特征分块处理
        self.attn0 = OverlapBlockAttention(channel, num_heads, block_size, overlap)
        self.attn1 = OverlapBlockAttention(channel, num_heads, block_size, overlap)
        self.attn2 = OverlapBlockAttention(channel, num_heads, block_size, overlap)

        # 使用 Mlp 类来融合特征
        self.mlp = Mlp(in_features=channel * 3 * 320 * 320, hidden_features=channel, out_features=channel * 320 * 320)

    def forward(self, x):
        # 获取不同尺度的特征图
        p3, p4, p5 = x[0], x[1], x[2]
        # 对每个尺度特征图进行卷积，统一通道数
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        # 上采样到与 p3 相同的尺寸
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        # 同样上采样到 p3 的尺寸
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')

        # 对不同尺度特征分别应用分块注意力机制
        p3 = self.attn0(p3)
        p4_2 = self.attn1(p4_2)
        p5_2 = self.attn2(p5_2)

        # 合并特征并展平为一维向量
        combined_features = torch.cat([p3, p4_2, p5_2], dim=1)
        combined_features = combined_features.view(combined_features.size(0), -1)  # 展平

        # 通过 Mlp 处理特征
        x = self.mlp(combined_features)

        # 恢复形状，确保输出的通道数与设定的一致
        batch_size = combined_features.size(0)
        x = x.view(batch_size, self.channel, 320, 320)  # 使用 self.channel

        return x




class Add(nn.Module):
    def __init__(self, ch=256):
        super().__init__()

    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class attention_model(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x
