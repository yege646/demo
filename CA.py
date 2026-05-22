class CoordAtt(nn.Module):
    def __init__(self, c1, reduction=32):
        super().__init__()
        reduced_c = max(8, c1 // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c1, reduced_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_c)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(reduced_c, c1, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced_c, c1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = self.pool_h(x)               # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)   # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        return x * self.sigmoid(self.conv_h(y_h)) * self.sigmoid(self.conv_w(y_w))