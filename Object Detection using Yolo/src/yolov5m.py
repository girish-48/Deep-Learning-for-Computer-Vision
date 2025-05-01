import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Common Modules =====
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden_channels = int(c2 * e)
        self.cv1 = ConvBNAct(c1, hidden_channels, 1, 1)
        self.cv2 = ConvBNAct(hidden_channels, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden_channels = int(c2 * e)
        self.cv1 = ConvBNAct(c1, hidden_channels, 1, 1)
        self.cv2 = ConvBNAct(c1, hidden_channels, 1, 1)
        self.cv3 = ConvBNAct(2 * hidden_channels, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x, self.m(x), self.m(self.m(x)), self.m(self.m(self.m(x)))], 1))

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.d = dim

    def forward(self, x):
        return torch.cat(x, self.d)

# ===== YOLOv5 Backbone + Neck + Head =====
class YOLOv5FCN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        self.layer0 = ConvBNAct(3, 64, 6, 2, 2)      # P1/2
        self.layer1 = ConvBNAct(64, 128, 3, 2)       # P2/4
        self.layer2 = C3(128, 128, n=3)
        self.layer3 = ConvBNAct(128, 256, 3, 2)      # P3/8
        self.layer4 = C3(256, 256, n=6)
        self.layer5 = ConvBNAct(256, 512, 3, 2)      # P4/16
        self.layer6 = C3(512, 512, n=9)
        self.layer7 = ConvBNAct(512, 1024, 3, 2)     # P5/32
        self.layer8 = C3(1024, 1024, n=3)
        self.layer9 = SPPF(1024, 1024, k=5)

        # Neck
        self.layer10 = ConvBNAct(1024, 512, 1, 1)
        self.layer11 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer12 = Concat()
        self.layer13 = C3(1024, 512, n=3, shortcut=False)

        self.layer14 = ConvBNAct(512, 256, 1, 1)
        self.layer15 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer16 = Concat()
        self.layer17 = C3(512, 256, n=3, shortcut=False)

        self.layer18 = ConvBNAct(256, 256, 3, 2)
        self.layer19 = Concat()
        self.layer20 = C3(512, 512, n=3, shortcut=False)

        self.layer21 = ConvBNAct(768, 512, 3, 2)
        self.layer22 = Concat()
        self.layer23 = C3(1536, 1024, n=3, shortcut=False)

        # Output (P3/8, P4/16, P5/32)
        self.detect = nn.ModuleList([
            nn.Conv2d(256, (5 + num_classes) * 3, 1),
            nn.Conv2d(512, (5 + num_classes) * 3, 1),
            nn.Conv2d(1024, (5 + num_classes) * 3, 1),
        ])

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)

        x10 = self.layer10(x9)
        x11 = self.layer11(x10)
        x12 = self.layer12([x11, x6])
        x13 = self.layer13(x12)

        x14 = self.layer14(x13)
        x15 = self.layer15(x14)
        x16 = self.layer16([x15, x4])
        x17 = self.layer17(x16)

        x18 = self.layer18(x17)
        x19 = self.layer19([x18, x13])
        x20 = self.layer20(x19)

        x21 = self.layer21(x20)
        x22 = self.layer22([x21, x10])
        x23 = self.layer23(x22)

        out_small  = self.detect[0](x17)
        out_medium = self.detect[1](x20)
        out_large  = self.detect[2](x23)

        return [out_small, out_medium, out_large]


if __name__ == '__main__':
    model = YOLOv5FCN(num_classes=20)  # for Pascal VOC
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    for out in y:
        print(out.shape)  # should be (1, 75, H, W) where 75 = 3 anchors * (5 + 20 classes)