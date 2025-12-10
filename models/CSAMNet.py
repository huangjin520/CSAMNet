import re
import torch.nn.functional as F
import torch.nn as nn
from torch import nn
import torch
import torch.nn.functional
from dropblock import LinearScheduler, DropBlock2D

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        rate1, rate2, rate3 = tuple(atrous_rates)

        out_channels = int(in_channels / 2)

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        # 全局平均池化
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

        self.dim_reduction = Conv3Relu(out_channels * 5, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        feat4 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)

        out = self.dim_reduction(torch.cat((feat0, feat1, feat2, feat3, feat4), 1))

        return out


class DropBlock(nn.Module):
    """
    [Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks
    """
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )

    def forward(self, feats: list):
        if self.training:  # 只在训练的时候加上dropblock
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplanel,inplaneh, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1,x2):
        low_feature=x1
        h_feature = x2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.0003):
        super().__init__()
        inter_channels = in_channels // 4


        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )


        self.classify = nn.Conv2d(in_channels=inter_channels, out_channels= out_channels, kernel_size=1,
                                        stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
       
        x = self.last_conv(x)
        pred = self.classify(x)
        return pred

class Conv1Relu(nn.Module):  # 1*1卷积用来降维
    def __init__(self, in_ch, out_ch):
        super(Conv1Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

    
class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class CoA(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class LayerNorm(nn.Module):
    """ From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class CA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
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
        self.conv1_1_1 = BasicConv2d(input_dim // 2, embed_dim, 1)
        self.ca_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.coa_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.ca = CA(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)
        self.coa = CoA(in_channels=embed_dim,out_channels=embed_dim)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1) 
        ca = self.ca(self.ca_11conv(x_0)) 
        coa = self.coa(self.coa_11conv(x_1)) 
        x = torch.cat([ca, coa], dim=1)
        x = self.conv1_1(x)

        return x

def fusion_block(input_dim,output_dim):
    block = CSAM(input_dim, output_dim)
    return block

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )

class attention_fusion(nn.Module):
    def __init__(self,dim):
        super(attention_fusion,self).__init__()
        self.attention = fusion_block(dim*4,dim)
    def forward(self,feature_list):
        for i,feature in enumerate(feature_list[:-1]):
            feature = F.pixel_shuffle(feature,1)
            x = feature if i ==0 else torch.cat([x,feature],dim=1)
        x = torch.cat([x,feature_list[-1]],dim=1)
        x = self.attention(x)
        return x
    
class FPNNeck(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 1, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.scn41= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn31= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn21= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.final_Conv5 = Conv3Relu(inplanes , inplanes)

        self.fusion_module = attention_fusion(inplanes)

        self.expand_field = ASPP(inplanes * 8)

        self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   # 降维
        self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
        self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)

        self.final_Conv = Conv3Relu(inplanes * 4, inplanes)
            

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4 = ms_feats

        [fa1, fa2, fa3, fa4] = self.drop([fa1, fa2, fa3, fa4])  # dropblock

        feature1 = self.stage1_Conv1(torch.cat([fa1], 1))  # inplanes
        feature2 = self.stage2_Conv1(torch.cat([fa2], 1))  # inplanes * 2
        feature3 = self.stage3_Conv1(torch.cat([fa3], 1))  # inplanes * 4
        feature4 = self.stage4_Conv1(torch.cat([fa4], 1))  # inplanes * 8
        if self.expand_field is not None:
            feature4 = self.expand_field(feature4)
       
        feature3_2 = self.stage4_Conv_after_up(self.up(feature4))
        feature3 = self.stage3_Conv2(torch.cat([feature3, feature3_2], 1))

        feature2_2 = self.stage3_Conv_after_up(self.up(feature3))
        feature2 = self.stage2_Conv2(torch.cat([feature2, feature2_2], 1))

        feature1_2 = self.stage2_Conv_after_up(self.up(feature2))
        feature1 = self.stage1_Conv2(torch.cat([feature1, feature1_2], 1))
        
        feature4=self.scn41(feature1, self.stage4_Conv3(feature4))
        feature3=self.scn31(feature1, self.stage3_Conv3(feature3))
        feature2=self.scn21(feature1, self.stage2_Conv3(feature2))
        feature = self.fusion_module([feature1,feature2,feature3,feature4])

        return feature

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

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
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x)) #(128, 16, 16)
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
        return x

class DC(nn.Module):
    def __init__(self, dim=36):
        super(DC, self).__init__()
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
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),  mode='nearest')
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
    """ U-Net downsampling block. """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        # layers: optional downsampling, 2 x (conv + bn + relu)
        self.maxpool = nn.MaxPool2d((2, 2)) if downsample else None
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class UnetEncoder(nn.Module):
    """U-Net encoder. https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(self, pretrained=False, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        if pretrained == True:
            print("WARNING: No pre-trained model available for U-Net encoder!")
        self.module1 = UnetDownModule(3, self.embed_dim*1)
        self.module2 = UnetDownModule(self.embed_dim*1, self.embed_dim*2)
        self.module3 = UnetDownModule(self.embed_dim*2, self.embed_dim*4)
        self.module4 = UnetDownModule(self.embed_dim*4, self.embed_dim*8)

    def forward(self, x):
        x1 = self.module1(x)
        x2 = self.module2(x1)
        x3 = self.module3(x2)
        x4 = self.module4(x3)

        return x1, x2, x3, x4

class Seg_Detection(nn.Module):
    def __init__(self, backbone, pretrain=''):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", backbone.split("_")[-1]))  # backbone的名称中必须在"_"之后加上它的通道数
        self.backbone = UnetEncoder(embed_dim= self.inplanes)  
        self._create_neck()
        self._create_heads()
        
        self.block1 = DCF(self.inplanes)
        self.block2 = DCF(self.inplanes * 2)
        self.block3 = DCF(self.inplanes * 4)
        self.block4 = DCF(self.inplanes * 8)

        if pretrain.endswith(".pt"):
            self._init_weight(pretrain)   # todo:这里预训练初始化和 hrnet主干网络的初始化有冲突，必须要改！


    def forward(self, x):
        _, _, h_input, w_input = x.shape
        f1, f2, f3, f4 = self.backbone(x)  # feature_a_1: 输入图像a的最大输出特征图
        ms_feats = self.block1(f1), self.block2(f2), self.block3(f3), self.block4(f4)  # 多尺度特征 
        feature = self.neck(ms_feats)
        out = self.head_forward(feature , out_size=(h_input, w_input))
        return out

    def head_forward(self, feature , out_size):

        out = F.interpolate(self.head(feature ), size=out_size, mode='bilinear', align_corners=True)
      
        return out
  
    def _create_neck(self):
        self.neck = FPNNeck(self.inplanes)

    def _create_heads(self):
        self.head = FCNHead(self.inplanes, 2)

class CSAMNet(nn.Module):
    def __init__(self,
                 backbone: str = "unet_32",
                 device: str | torch.device | None = None,
                 eval_mode: bool = False):
        super().__init__()
        self.model = Seg_Detection(
            backbone=backbone
        )
        if device is not None:
            self.model = self.model.to(device)
        if eval_mode:
            self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    # torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    model = CSAMNet(backbone="unet_32").to(device)
    input = torch.randn(4,3,256,256).to(device)
    output = model(input)
    print(model)
    print(output.shape)