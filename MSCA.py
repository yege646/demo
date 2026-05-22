import torch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MSCA(nn.Module):
    """
    Morphology-Sensitive Coordinate Attention (MSCA) - Improved
    改进版：解决信号衰减问题，扩大形态感受野，降低计算量。
    """

    def __init__(self, inp, reduction=32):
        super(MSCA, self).__init__()

        mip = max(8, inp // reduction)

        # === 1. 改进的形态感知分支 (Morphology Branch) ===
        # 改进点：
        # 1. 使用 5x5 Kernel 扩大感受野，捕捉更大的花瓣/花蕊结构。
        # 2. 使用 Depthwise Convolution (groups=mip) 降低参数量，专注提取形状而非通道融合。
        self.morph_conv = nn.Sequential(
            # 降维
            nn.Conv2d(inp, mip, kernel_size=1, bias=False),
            nn.BatchNorm2d(mip),
            h_swish(),
            # 深度卷积提取形态 (Depthwise 5x5)
            nn.Conv2d(mip, mip, kernel_size=5, stride=1, padding=2, groups=mip, bias=False),
            nn.BatchNorm2d(mip),
            h_swish(),
            # 升维
            nn.Conv2d(mip, inp, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # === 2. 坐标编码分支 (Coordinate Branch) ===
        # 保持原有的 CA 逻辑，这部分在定位任务中非常有效
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # --- 分支 1: 形态注意力 (Morphology Attention) ---
        # 提取花卉的边缘和局部形状信息
        morph_attn = self.morph_conv(x)

        # --- 分支 2: 坐标注意力 (Coordinate Attention) ---
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 合并坐标注意力
        coord_attn = a_w * a_h

        # --- 特征融合 (Fusion Strategy) ---
        # 改进点：使用加法融合或混合融合，防止连乘导致的特征消失
        # 逻辑：特征 = 原始特征 * (坐标关注 + 形态关注)
        # 意义：如果"位置对了" OR "形状像花瓣"，都应该激活该特征

        attn_fusion = (coord_attn + morph_attn) / 2.0  # 平均化，保持数值稳定

        # 或者使用更激进的： attn_fusion = torch.max(coord_attn, morph_attn)

        out = identity * attn_fusion

        return out