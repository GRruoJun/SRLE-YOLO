import torch #4 ，3d卷积改成自注意力,严重爆显存
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
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
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
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        # s = self.conv_s_post_up(s)
        lms = torch.cat([l, m, s], dim=1)
        return lms



        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # # 打印输入张量的形状
        # print(f"Input shape: {x.shape}")

        # 计算查询、键和值
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width).transpose(2, 3)  # [B, H, HW, C//H]
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width).transpose(2, 3)  # [B, H, HW, C//H]
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width).transpose(2, 3)  # [B, H, HW, C//H]

        # # 打印中间张量的形状
        # print(f"Query shape: {query.shape}")
        # print(f"Key shape: {key.shape}")
        # print(f"Value shape: {value.shape}")

        # 计算注意力
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [B, H, HW, HW]
        attention = F.softmax(attention, dim=-1)  # [B, H, HW, HW]

        # # 打印注意力张量的形状
        # print(f"Attention shape: {attention.shape}")

        # 进行注意力乘法
        out = torch.matmul(attention, value)  # [B, H, HW, C//H]
        out = out.transpose(2, 3).contiguous().view(batch_size, channels, height, width)  # [B, C, H, W]

        # 通过一个卷积层输出
        out = self.out_conv(out)

        return out + x  # 残差连接


    
class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)

        # 自注意力模块
        self.self_attention = MultiHeadSelfAttention(channel)

        # 批归一化、激活和池化层
        self.bn = nn.BatchNorm2d(channel * 3)  # 这里假设拼接后的通道数为 channel * 3
        self.act = nn.ReLU()  # 可根据需要更改激活函数
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1))  # 选择适合的池化层

    def forward(self, x):
        # 获取三个不同尺度的特征图
        p3, p4, p5 = x[0], x[1], x[2]

        # 对每个尺度的特征图进行卷积
        p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p5_2 = self.conv2(p5)

        # 将 p4 和 p5 插值缩放到与 p3 一致的尺寸
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')

        # 在拼接之前应用自注意力
        p3_att = self.self_attention(p3)
        p4_att = self.self_attention(p4_2)
        p5_att = self.self_attention(p5_2)

        # 将处理后的特征图拼接
        combine = torch.cat([p3_att, p4_att, p5_att], dim=1)

        # 批归一化和激活
        bn = self.bn(combine)
        act = self.act(bn)

        # 池化操作
        x = self.pool(act)
        x = torch.squeeze(x, 2)  # 去掉尺寸为1的维度

        return x

class Add(nn.Module):
    # Concatenate a list of tensors along dimension
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
    # Concatenate a list of tensors along dimension
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