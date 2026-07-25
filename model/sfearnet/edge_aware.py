import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F
from .semantic_flow import Semantic_flow
from .CBAM import CBAM
from torch import Tensor


class Doubleconv(nn.Module):
    def __init__(self, input_channels, num_channels,):
        super().__init__()
        # self.conv1 = nn.Conv2d(input_channels, num_channels,
        #                        kernel_size=3, padding=1, stride=strides),
        # self.conv1=nn.Sequential(
        #     nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides),
        #     nn.BatchNorm2d(num_channels),
        #     nn.ReLU()
        #
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3,
                      padding=1, groups=input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # self.conv2 = nn.Conv2d(num_channels, num_channels,
        #                        kernel_size=3, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3,
                      padding=1, groups=num_channels),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.conv2(Y)
        return Y


class Edge_Aware_1(nn.Module):
    def __init__(self, channel1, channel2):
        super(Edge_Aware_1, self).__init__()
        # self.conv_channel1=nn.Sequential(
        #     nn.Conv2d(channel1,channel1//4,kernel_size=1),
        #     nn.BatchNorm2d(channel1//4),
        #     nn.ReLU
        # )
        self.conv_channel1 = nn.Sequential(
            nn.Conv2d(channel1, channel1, kernel_size=3,
                      padding=1, groups=channel1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(),
            nn.Conv2d(channel1, channel1, kernel_size=1),
            nn.BatchNorm2d(channel1),
            nn.ReLU()
        )
        self.conv_channel2 = nn.Sequential(
            nn.Conv2d(channel2, channel2, kernel_size=3,
                      padding=1, groups=channel2),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.Conv2d(channel2, channel2, kernel_size=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU()
        )

        self.doubleconv = Doubleconv(
            channel1 + channel2, (channel1 + channel2)*2)
        self.conv1 = nn.Sequential(
            nn.Conv2d((channel1 + channel2)*2, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            # nn.Sigmoid()
        )
        self.sef = Semantic_flow(512, 64)

    def forward(self, input1, input4):
        # print(input1.size(),input4.size())
        input1 = self.conv_channel1(input1)
        input4 = self.conv_channel2(input4)
        # print(input1.size(),input4.size())
        input4_sef = self.sef(input4, input1)
        input4_big = F.interpolate(input4, size=input1.size(
        )[2:], mode='bilinear', align_corners=False)+input4_sef
        input = torch.cat([input4_big, input1], dim=1)
        input_res = self.doubleconv(input)
        out = self.conv1(input_res)
        return out


class Edge_Aware_0(nn.Module):
    def __init__(self, channel1, channel2):
        super(Edge_Aware_0, self).__init__()
        # self.conv_channel1=nn.Sequential(
        #     nn.Conv2d(channel1,channel1//4,kernel_size=1),
        #     nn.BatchNorm2d(channel1//4),
        #     nn.ReLU
        # )
        self.conv_channel1 = nn.Sequential(
            nn.Conv2d(channel1, channel1, kernel_size=3,
                      padding=1, groups=channel1),
            nn.BatchNorm2d(channel1),
            nn.ReLU(),
            nn.Conv2d(channel1, channel1, kernel_size=1),
            nn.BatchNorm2d(channel1),
            nn.ReLU()
        )
        self.conv_channel2 = nn.Sequential(
            nn.Conv2d(channel2, channel2, kernel_size=3,
                      padding=1, groups=channel2),
            nn.BatchNorm2d(channel2),
            nn.ReLU(),
            nn.Conv2d(channel2, channel2, kernel_size=1),
            nn.BatchNorm2d(channel2),
            nn.ReLU()
        )

        self.doubleconv = Doubleconv(
            channel1 + channel2, (channel1 + channel2)*2)
        self.conv1 = nn.Sequential(
            nn.Conv2d((channel1 + channel2)*2, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            # nn.Sigmoid()
        )

        self.sef = Semantic_flow(256, 32)#【B, C4, H1, W1】

    def forward(self,
                input1: Tensor,
                input4: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        input1 : Tensor
            (C1, H1, W1)
        input4 : Tensor
            (C4, H4, W4)

        Returns
        -------
        Tensor
            _description_
        """

        input1 = self.conv_channel1(input1)  # (channel1, H1, W1)与输入相同，保持通道数不变
        input4 = self.conv_channel2(input4)  # (channel4, H4, W4)与输入相同，保持通道数不变
                
        input4_sef = self.sef(input4, input1) #【B, C4, H1, W1】将input4和input1输入到语义流模块中，得到对齐后的input4特征图input4_sef
        
        input4_big = F.interpolate(input4, size=input1.size()[2:], 
                                   mode='bilinear', 
                                   align_corners=False)+input4_sef#【B, C4, H1, W1】将input4进行双线性上采样到input1的高宽，并与input4_sef相加，得到input4_big
        # print(input4_big.size())
        input = torch.cat([input4_big, input1], dim=1)#【B, C4+C1, H1, W1】将input4_big和input1在通道维度上进行拼接，得到input
        input_res = self.doubleconv(input)#【B, (C4+C1)*2, H1, W1】将input输入到doubleconv中，得到input_res
        out = self.conv1(input_res)#【B, 2, H1, W1】将input_res输入到conv1中，得到out
        return out#【B, 2, H1, W1】


class Edge_Guidance_1(nn.Module):
    def __init__(self):
        super(Edge_Guidance_1, self).__init__()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.ea = Edge_Aware_1(64, 512)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.ca1 = CBAM(64)
        self.ca2 = CBAM(128)
        self.ca3 = CBAM(320)
        self.ca4 = CBAM(512)

    def forward(self, input):
        input1, input2, input3, input4 = input
        edge_64_2 = self.ea(input1, input4)
        edge_64 = self.conv2_1(edge_64_2)
        # print(edge_64_1.size())
        edge_32 = self.maxpool1(edge_64)
        edge_16 = self.maxpool2(edge_32)
        edge_8 = self.maxpool3(edge_16)

        feature_1 = input1*edge_64+input1
        feature_2 = input2*edge_32+input2
        feature_3 = input3*edge_16+input3
        feature_4 = input4*edge_8+input4
        # print(feature_1.size(),feature_2.size(),feature_3.size(),feature_4.size())

        feature_1 = self.ca1(feature_1)
        feature_2 = self.ca2(feature_2)
        feature_3 = self.ca3(feature_3)
        feature_4 = self.ca4(feature_4)

        return edge_64_2, feature_1, feature_2, feature_3, feature_4


class Edge_Guidance_0(nn.Module):
    def __init__(self):
        super(Edge_Guidance_0, self).__init__()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.ea = Edge_Aware_0(32, 256)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.ca1 = CBAM(32)
        self.ca2 = CBAM(64)
        self.ca3 = CBAM(160)
        self.ca4 = CBAM(256)

    def forward(self, input):
        input1, input2, input3, input4 = input#输入的四个特征图，分别是输入图像的1/4、1/8、1/16、1/32分辨率的特征图，通道数分别是32、64、160、256（b0），并且经过了差异增强处理-lgx
        edge_64_2 = self.ea(input1, input4)#得到边缘信息，【B, 2, H1, W1】-lgx
        edge_64 = self.conv2_1(edge_64_2)#【B, 1, H1, W1】将边缘信息输入到conv2_1中，得到edge_64
        # print(edge_64_1.size())
        edge_32 = self.maxpool1(edge_64)#【B, 1, H2, W2】将edge_64进行最大池化，得到edge_32
        edge_16 = self.maxpool2(edge_32)#【B, 1, H3, W3】将edge_32进行最大池化，得到edge_16
        edge_8 = self.maxpool3(edge_16)#【B, 1, H4, W4】将edge_16进行最大池化，得到edge_8

        feature_1 = input1*edge_64+input1#【B, C1, H1, W1】将input1和edge_64进行逐元素乘法，再加上input1，得到feature_1，增强了边缘信息的特征图
        feature_2 = input2*edge_32+input2#【B, C2, H2, W2】将input2和edge_32进行逐元素乘法，再加上input2，得到feature_2，增强了边缘信息的特征图
        feature_3 = input3*edge_16+input3#【B, C3, H3, W3】将input3和edge_16进行逐元素乘法，再加上input3，得到feature_3，增强了边缘信息的特征图
        feature_4 = input4*edge_8+input4#【B, C4, H4, W4】将input4和edge_8进行逐元素乘法，再加上input4，得到feature_4，增强了边缘信息的特征图

        feature_1 = self.ca1(feature_1)#【B, C1, H1, W1】将feature_1输入到ca1中，得到增强后的feature_1
        feature_2 = self.ca2(feature_2)#【B, C2, H2, W2】将feature_2输入到ca2中，得到增强后的feature_2
        feature_3 = self.ca3(feature_3)#【B, C3, H3, W3】将feature_3输入到ca3中，得到增强后的feature_3
        feature_4 = self.ca4(feature_4)#【B, C4, H4, W4】将feature_4输入到ca4中，得到增强后的feature_4

        return edge_64_2, feature_1, feature_2, feature_3, feature_4#返回边缘信息和增强后的四个特征图【B, 2, H1, W1】,【B, C1, H1, W1】,【B, C2, H2, W2】,【B, C3, H3, W3】,【B, C4, H4, W4】
