# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from typing import Tuple

# from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .backbone import mit_b0, mit_b1

from .semantic_flow import Semantic_flow, Resampler
from .edge_aware import Edge_Guidance_1, Edge_Guidance_0
from .pyramid import Pyramid_Merge


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead_1(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead_1, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.sef1 = Semantic_flow(256, 64)
        self.sef2 = Semantic_flow(256, 64)
        self.sef3 = Semantic_flow(256, 64)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # b0:torch.Size([8, 32, 64, 64]) torch.Size([8, 64, 32, 32]) torch.Size([8, 160, 16, 16]) torch.Size([8, 256, 8, 8])
        # print(c1.size(), c2.size(), c3.size(), c4.size())
        # b1 torch.Size([8, 64, 64, 64]) torch.Size([8, 128, 32, 32]) torch.Size([8, 320, 16, 16]) torch.Size([8, 512, 8, 8])
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        # print(_c4.size())
        _c4_1 = self.sef1(_c4, c1)
        # print(_c4.size())
        _c4_2 = F.interpolate(_c4, size=c1.size()[
                              2:], mode='bilinear', align_corners=False)
        # print(_c4.size())
        _c4 = _c4_1 + _c4_2

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])

        _c3_1 = self.sef2(_c3, c1)
        _c3_2 = F.interpolate(_c3, size=c1.size()[
                              2:], mode='bilinear', align_corners=False)
        # print(_c3.size())
        _c3 = _c3_1 + _c3_2

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        _c2_1 = self.sef3(_c2, c1)
        _c2_2 = F.interpolate(_c2, size=c1.size()[
                              2:], mode='bilinear', align_corners=False)
        _c2 = _c2_1 + _c2_2
        # print(_c2.size())

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print(x.size())
        return x


class SegFormerHead_0(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead_0, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.sef1 = Semantic_flow(256, 32)
        self.sef2 = Semantic_flow(256, 32)
        self.sef3 = Semantic_flow(256, 32)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # torch.Size([8, 32, 64, 64]) torch.Size([8, 64, 32, 32]) torch.Size([8, 160, 16, 16]) torch.Size([8, 256, 8, 8])
        # print(c1.size(), c2.size(), c3.size(), c4.size())
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3])
        _c4_1 = self.sef1(_c4, c1)
        # print(_c4.size())
        _c4_2 = F.interpolate(_c4, size=c1.size()[
                              2:], mode='bilinear', align_corners=False)
        # print(_c4.size())
        _c4 = _c4_1 + _c4_2

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3])

        _c3_1 = self.sef2(_c3, c1)
        _c3_2 = F.interpolate(_c3, size=c1.size()[
                              2:], mode='bilinear', align_corners=False)
        # print(_c3.size())
        _c3 = _c3_1 + _c3_2

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3])
        # print(_c2.size())
        _c2_1 = self.sef3(_c2, c1)
        _c2_2 = F.interpolate(_c2, size=c1.size()[
                              2:], mode='bilinear', align_corners=False)
        _c2 = _c2_1 + _c2_2
        # print(_c2.size())

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # print(x.size())
        return x


class SFEARNet(nn.Module):
    def __init__(self, num_classes: int = 21, phi: str = 'b1', pretrained: bool = False) -> None:
        """
        用于双时相遥感变化检测的 SFEARNet 主模型。

        Args:
            num_classes (int): 变化分支和边缘分支的类别数。
                在本项目中通常设置为 2（unchanged/changed）。
            phi (str): 主干规模标识。当前支持 'b0' 与 'b1'。
            pretrained (bool): 是否加载主干预训练权重。
        """
        super(SFEARNet, self).__init__()
        self.in_channels = {
            # 'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            # 'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            # 'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            # 'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
            'b0': mit_b0, 'b1': mit_b1,
        }[phi](pretrained)
        self.embedding_dim = {
            # 'b0': 256, 'b1': 256, 'b2': 768,
            # 'b3': 768, 'b4': 768, 'b5': 768,
            'b0': 256, 'b1': 256,
        }[phi]

        if phi == 'b1':
            self.decode_head = SegFormerHead_1(
                num_classes, self.in_channels, self.embedding_dim)
            self.re = Resampler(64, 256)
            self.re2 = Resampler(64, 256)
            self.eg = Edge_Guidance_1()
            self.py1 = Pyramid_Merge(64)
            self.py2 = Pyramid_Merge(128)
            self.py3 = Pyramid_Merge(320)
            self.py4 = Pyramid_Merge(512)
        elif phi == 'b0':
            self.decode_head = SegFormerHead_0(
                num_classes, self.in_channels, self.embedding_dim)
            self.re = Resampler(64, 256)
            self.re2 = Resampler(64, 256)
            self.eg = Edge_Guidance_0()
            self.py1 = Pyramid_Merge(32)
            self.py2 = Pyramid_Merge(64)
            self.py3 = Pyramid_Merge(160)
            self.py4 = Pyramid_Merge(256)

    def forward(
        self, input1: torch.Tensor, input2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SFEARNet 前向传播。

        输入来自项目数据集与 DataLoader：
        图像张量为 ``[B, 3, H, W]``，标签/边缘为 ``[B, H, W]``。
        该方法消费双时相图像，返回用于分割、边缘监督和对比监督的四个张量。

        Args:
            input1 (torch.Tensor): 时相 A 图像，形状 ``[B, 3, H, W]``。
            input2 (torch.Tensor): 时相 B 图像，形状 ``[B, 3, H, W]``。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (x, edge, feat1, feat2), where:
                - x: 变化预测 logits，形状 ``[B, num_classes, H, W]``。
                - edge: 边缘预测 logits，形状 ``[B, num_classes, H, W]``。
                - feat1: 时相 A 深层特征（上采样后），形状 ``[B, C_deep, H, W]``。
                - feat2: 时相 B 深层特征（上采样后），形状 ``[B, C_deep, H, W]``。

        Note:
            ``C_deep`` 由主干规模决定：
            ``phi='b0'`` 时为 256，``phi='b1'`` 时为 512。
        """
        if not hasattr(self, '_printed_io_shapes'):
            print(
                f"SFEARNet input shapes: input1={tuple(input1.shape)}, input2={tuple(input2.shape)}")
            self._printed_io_shapes = True

        H, W = input1.size(2), input1.size(3)

        # 返回的是4个特征图的列表，由MixVisionTransformer返回，分别是输入图像的1/4、1/8、1/16、1/32分辨率的特征图，通道数分别是32、64、160、256（b0）或64、128、320、512（b1）
        x1 = self.backbone(input1)
        x2 = self.backbone(input2)  # 返回的是4个特征图的列表
        # x=[torch.abs(xa-xb) for xa,xb in zip(x1,x2)]
        x_0 = self.py1(x1[0], x2[0])
        x_1 = self.py2(x1[1], x2[1])
        x_2 = self.py3(x1[2], x2[2])
        x_3 = self.py4(x1[3], x2[3])
        x = [x_0, x_1, x_2, x_3]
        # print(x_3.size())

        # edge_64_2,feature_1,feature_2,feature_3,feature_4=eg(x),得到两类，一个是边缘信息，另外一类是对PE提取后的特征图进行边缘引导增强后的特征图，分别是feature_1、feature_2、feature_3、feature_4，对应输入的四个特征图。
        eg = self.eg(x)
        edge = eg[0]#边缘信息-lgx
        x = eg[1:]
        # print(x[0].size())
        x = self.decode_head(x)#形状为 [B, num_classes, H_c1, W_c1] 的 logits（与最高分辨率特征 c1 对齐）-lgx

        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.re(x)#上采样到输入图像大小-lgx
        edge = self.re2(edge)

        # Extract features for contrastive loss
        feat1 = F.interpolate(x1[3], size=(
            H, W), mode='bilinear', align_corners=False)
        feat2 = F.interpolate(x2[3], size=(
            H, W), mode='bilinear', align_corners=False)

        if hasattr(self, '_printed_io_shapes') and self._printed_io_shapes:
            print(
                f"SFEARNet output shapes: x={tuple(x.shape)}, edge={tuple(edge.shape)}, feat1={tuple(feat1.shape)}, feat2={tuple(feat2.shape)}")
            self._printed_io_shapes = False
        #预测结果，边缘，输入1的深层特征图，输入2的深层特征图
        return x, edge, feat1, feat2


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input1 = torch.rand([8, 3, 256, 256]).to(device)
    input2 = torch.rand([8, 3, 256, 256]).to(device)
    model = SFEARNet(num_classes=2, phi='b0', pretrained=False).to(device)
    out = model(input1, input2)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    print(out[3].size())

    input_data1 = torch.randn([1, 3, 256, 256]).to(device)
    input_data2 = torch.randn([1, 3, 256, 256]).to(device)
    flops, params = profile(model, inputs=(input_data1, input_data2,))
    print('Number of parameters: ' + str(params))
    print('FLOPs: ' + str(flops))
    print(params / 10 ** 6)
    print(flops / 10 ** 9)

# Number of parameters: 5563101.0
# FLOPs: 4645815552.0
# 5.563101
# 4.645815552

# Number of parameters: 20885663.0
# FLOPs: 14775373056.0
# 20.885663
# 14.775373056
