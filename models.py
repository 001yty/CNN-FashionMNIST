import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：DW 3x3 + PW 1x1。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


def make_norm(norm_type, num_channels):
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        # GroupNorm 的 groups 必须整除 channels，这里统一用 8 组
        # 本实验通道为 32/64/128，均能被 8 整除
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown norm: {norm_type}")


def make_act(act_type):
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown act: {act_type}")


class ConvBlock(nn.Module):
    """Conv -> Norm -> Act -> (SE)"""

    def __init__(self, in_ch, out_ch, conv_type="standard",
                 use_se=False, act_type="relu", norm_type="bn"):
        super().__init__()
        if conv_type == "standard":
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        elif conv_type == "dw":
            self.conv = DepthwiseSeparableConv(in_ch, out_ch, 3, padding=1)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.norm = make_norm(norm_type, out_ch)
        self.act = make_act(act_type)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return x


class SmallCNN(nn.Module):
    """3 层卷积的小型 CNN，通道 32 -> 64 -> 128，GAP + FC。"""

    def __init__(self, conv_type="standard", use_se=False,
                 act_type="relu", norm_type="bn", num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, conv_type, use_se, act_type, norm_type),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, conv_type, use_se, act_type, norm_type),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, conv_type, use_se, act_type, norm_type),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)