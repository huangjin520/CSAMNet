import re

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dropblock import DropBlock2D, LinearScheduler
except ImportError:
    DropBlock2D = None
    LinearScheduler = None


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18)):
        super().__init__()
        rate1, rate2, rate3 = tuple(atrous_rates)
        out_channels = in_channels // 2

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dim_reduction = Conv3Relu(out_channels * 5, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)
        return self.dim_reduction(torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1))


class DropBlock(nn.Module):
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()
        if DropBlock2D is None or LinearScheduler is None:
            self.drop = None
        else:
            self.drop = LinearScheduler(
                DropBlock2D(block_size=size, drop_prob=0.0),
                start_value=0.0,
                stop_value=rate,
                nr_steps=step,
            )

    def forward(self, feats):
        if self.training and self.drop is not None:
            return [self.drop(f) for f in feats]
        return feats

    def step(self):
        if self.drop is not None:
            self.drop.step()


class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.extract(x)


class AlignedModulev2PoolingAtten(nn.Module):
    def __init__(self, inplanel, inplaneh, outplane, kernel_size=3):
        super().__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        low_feature = x1
        h_feature = x2
        h_feature_origin = h_feature

        h, w = low_feature.shape[2:]
        size = (h, w)

        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], dim=1))
        flow_up, flow_down = flow[:, :2], flow[:, 2:]

        h_feature_warp = self.flow_warp(h_feature_origin, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1, keepdim=True)
        l_feature_mean = torch.mean(low_feature, dim=1, keepdim=True)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(
            torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], dim=1)
        )
        return h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

    @staticmethod
    def flow_warp(inputs, flow, size):
        out_h, out_w = size
        n, _, _, _ = inputs.size()

        norm = torch.tensor([[[[out_w, out_h]]]], dtype=inputs.dtype, device=inputs.device)
        h_grid = torch.linspace(-1.0, 1.0, out_h, dtype=inputs.dtype, device=inputs.device).view(-1, 1).repeat(1, out_w)
        w_grid = torch.linspace(-1.0, 1.0, out_w, dtype=inputs.dtype, device=inputs.device).repeat(out_h, 1)
        grid = torch.cat((w_grid.unsqueeze(2), h_grid.unsqueeze(2)), dim=2)
        grid = grid.repeat(n, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        return F.grid_sample(inputs, grid, align_corners=True)


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.0003):
        super().__init__()
        inter_channels = in_channels // 4
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inter_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(inter_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.classify = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.classify(self.last_conv(x))


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc2(F.relu(self.fc1(x1), inplace=True))
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc2(F.relu(self.fc1(x2), inplace=True))
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        return x.view(-1, self.input_channels, 1, 1)


class CoA(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super().__init__()
        assert in_channels == out_channels

        self.ca = ChannelAttention(
            input_channels=in_channels,
            internal_neurons=in_channels // channel_attention_reduce,
        )
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.act(self.conv(inputs))
        inputs = self.ca(inputs) * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv7_1(self.dconv1_7(x_init))
        x_2 = self.dconv11_1(self.dconv1_11(x_init))
        x_3 = self.dconv21_1(self.dconv1_21(x_init))

        x = x_init + x_1 + x_2 + x_3
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        return self.conv(out)


class CA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x

        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)

        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class CSAM(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.conv1_1 = BasicConv2d(embed_dim * 2, embed_dim, 1)
        self.ca_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.coa_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.ca = CA(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)
        self.coa = CoA(in_channels=embed_dim, out_channels=embed_dim)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        ca = self.ca(self.ca_11conv(x_0))
        coa = self.coa(self.coa_11conv(x_1))
        return self.conv1_1(torch.cat([ca, coa], dim=1))


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = CSAM(dim * 4, dim)

    def forward(self, feature_list):
        for i, feature in enumerate(feature_list[:-1]):
            x = feature if i == 0 else torch.cat([x, feature], dim=1)
        x = torch.cat([x, feature_list[-1]], dim=1)
        return self.attention(x)


class FPNNeck(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.stage1_conv1 = Conv3Relu(inplanes, inplanes)
        self.stage2_conv1 = Conv3Relu(inplanes * 2, inplanes * 2)
        self.stage3_conv1 = Conv3Relu(inplanes * 4, inplanes * 4)
        self.stage4_conv1 = Conv3Relu(inplanes * 8, inplanes * 8)

        self.stage2_conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage1_conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        self.scn41 = AlignedModulev2PoolingAtten(inplanes, inplanes, inplanes)
        self.scn31 = AlignedModulev2PoolingAtten(inplanes, inplanes, inplanes)
        self.scn21 = AlignedModulev2PoolingAtten(inplanes, inplanes, inplanes)

        self.fusion_module = AttentionFusion(inplanes)
        self.expand_field = ASPP(inplanes * 8)

        self.stage2_conv3 = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_conv3 = Conv3Relu(inplanes * 4, inplanes)
        self.stage4_conv3 = Conv3Relu(inplanes * 8, inplanes)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.drop = DropBlock(rate=0.15, size=7, step=30)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4 = self.drop(list(ms_feats))

        feature1 = self.stage1_conv1(fa1)
        feature2 = self.stage2_conv1(fa2)
        feature3 = self.stage3_conv1(fa3)
        feature4 = self.stage4_conv1(fa4)

        feature4 = self.expand_field(feature4)

        feature3_2 = self.stage4_conv_after_up(self.up(feature4))
        feature3 = self.stage3_conv2(torch.cat([feature3, feature3_2], dim=1))

        feature2_2 = self.stage3_conv_after_up(self.up(feature3))
        feature2 = self.stage2_conv2(torch.cat([feature2, feature2_2], dim=1))

        feature1_2 = self.stage2_conv_after_up(self.up(feature2))
        feature1 = self.stage1_conv2(torch.cat([feature1, feature1_2], dim=1))

        feature4 = self.scn41(feature1, self.stage4_conv3(feature4))
        feature3 = self.scn31(feature1, self.stage3_conv3(feature3))
        feature2 = self.scn21(feature1, self.stage2_conv3(feature2))

        return self.fusion_module([feature1, feature2, feature3, feature4])


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        return self.conv_1(x)


class Fusion(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)

        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim

    def forward(self, x):
        x = self.act(self.conv_0(x))
        if self.training:
            x1, x2 = torch.split(x, [self.p_dim, x.shape[1] - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = torch.cat([x1, x2], dim=1)
        else:
            x[:, : self.p_dim] = self.act(self.conv_1(x[:, : self.p_dim]))
        return self.conv_2(x)


class DC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)

        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(
            self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)),
            size=(h, w),
            mode="nearest",
        )

        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


class DCF(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.dc = DC(dim)
        self.fusion = Fusion(dim, ffn_scale)

    def forward(self, x):
        x = self.dc(F.normalize(x)) + x
        x = self.fusion(F.normalize(x)) + x
        return x


class UnetDownModule(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2) if downsample else None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class UnetEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.module1 = UnetDownModule(3, embed_dim)
        self.module2 = UnetDownModule(embed_dim, embed_dim * 2)
        self.module3 = UnetDownModule(embed_dim * 2, embed_dim * 4)
        self.module4 = UnetDownModule(embed_dim * 4, embed_dim * 8)

    def forward(self, x):
        x1 = self.module1(x)
        x2 = self.module2(x1)
        x3 = self.module3(x2)
        x4 = self.module4(x3)
        return x1, x2, x3, x4


class CSAMNet(nn.Module):
    def __init__(self, backbone: str = "unet_32"):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", backbone.split("_")[-1]))

        self.backbone = UnetEncoder(embed_dim=self.inplanes)
        self.neck = FPNNeck(self.inplanes)
        self.head = FCNHead(self.inplanes, 2)

        self.block1 = DCF(self.inplanes)
        self.block2 = DCF(self.inplanes * 2)
        self.block3 = DCF(self.inplanes * 4)
        self.block4 = DCF(self.inplanes * 8)

    def forward(self, x):
        _, _, h_input, w_input = x.shape

        f1, f2, f3, f4 = self.backbone(x)
        ms_feats = (self.block1(f1), self.block2(f2), self.block3(f3), self.block4(f4))

        feature = self.neck(ms_feats)
        out = self.head(feature)

        return F.interpolate(out, size=(h_input, w_input), mode="bilinear", align_corners=True)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CSAMNet(backbone="unet_32").to(device)
    inp = torch.randn(2, 3, 256, 256).to(device)
    out = model(inp)
    print(out.shape)
