import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F


class Semantic_flow(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Semantic_flow, self).__init__()
        # inchannel为低分辨率图片通道数 channel4=256，outchannel为高分辨率图片通道数channel1=32
        self.down_4 = nn.Conv2d(inchannel, outchannel, 1, bias=False)#(C1,H4,W4)
        self.down_1 = nn.Conv2d(outchannel, outchannel, 1, bias=False)#(C1,H1,W1)
        self.flow_make = nn.Conv2d(
            outchannel*2, 2, kernel_size=3, padding=1, bias=False)
        # self.conv=nn.Conv2d(inchannel*2,outchannel,1)

    def forward(self, feat_4, feat_1):
        """_summary_

        Parameters
        ----------
        feat_4: Tensor
            (C4, H4, W4) 低分辨率的high-level feature
        feat_1: Tensor
            (C1, H1, W1) 高分辨率的low-level feature

        Returns
        -------
        _type_
            _description_
        """
        # feat1 对应分辨率较高的特征图，feat_4即为低分辨率的 feature
        feat4_orign = feat_4#保存原始的feat_4，后续进行warp操作时需要用到
        h, w = feat_1.size()[2:]
        size = (h, w)#h1,w1
        
        # 将feat_4 和 feat_1 特征分别通过两个1x1卷积进行压缩，全部变成和C1相同的通道数-lgx
        feat_1 = self.down_1(feat_1)  # (C1, H1, W1)
        feat_4 = self.down_4(feat_4)  # (C1, H4, W4)
        
        # 将feat4 feature进行双线性上采样
        feat_4 = F.interpolate(feat_4,
                               size=size,
                               mode="bilinear",
                                  align_corners=False)  #将feat_4对齐到feat_1 (C1, H1, W1)
        
        # 预测语义流场 === 其实就是输入一个3x3的卷积
        # (Co*2, H1, W1) -> (2, H1, W1)
        flow = self.flow_make(torch.cat([feat_4, feat_1], 1))#[B, 2, H1, W1]得到边缘预测结果
        
        # 将Flow Field warp 到当前的 high-level feature中
        feat_4 = self.flow_warp(feat4_orign,#原始的feat_4，即低分辨率的high-level feature，【B, C4, H4, W4】
                                   flow,#预测的语义流场，【B, 2, H1, W1】
                                   size=size)#size-(H1, W1)
        return feat_4#【B, C4, H1, W1】

    @staticmethod
    def flow_warp(feat_4, flow, size):
        """
        TODO 让codex 概括出这个函数中每行的变量形状，以及输出的变量形状。

        Parameters
        ----------
        feat_4 : _type_
            (C4, H4, W4) 原始第四层特征，低分辨率的high-level feature
        flow : _type_
            (2, H1, W1)，预测结果
        size : _type_
            (H1, W1)

        Returns
        -------
        _type_
            _description_
        """
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸(H1,W1)
        n, c, h, w = feat_4.size()  # 对应低分辨率的high-level (B, C4, H4, W4) 特征图的尺寸

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(
            feat_4).to(feat_4.device)#归一化系数，【1,1,1,2】
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)#【H1, W1】生成h的转置矩阵
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)#【H1, W1】生成w的转置矩阵
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)#[H1, W1, 2]
        grid = grid.repeat(n, 1, 1, 1).type_as(feat_4).to(feat_4.device)#【H1, W1, 2】 -> [n, H1, W1, 2]
        grid = grid + flow.permute(0, 2, 3, 1) / norm#flow 先从 [n, 2, H1, W1] 变成 [n, H1, W1, 2]，归一化，再加上到基础上
        # print(grid.size())
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素。
        # 具体可以参考《Spatial Transformer Networks》，下方参考文献[2]
        output = F.grid_sample(feat_4, grid, align_corners=False)#【B, C4, H1, W1】
        # print(output.size())
        return output#【B, C4, H1, W1】


class Resampler(nn.Module):
    def __init__(self, input_size, output_size):
        super(Resampler, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, output_size), torch.linspace(-1, 1, output_size), indexing='ij')
        self.grid = torch.stack((grid_y, grid_x), 2).unsqueeze(0)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor : _type_
            [B, num_classes, H1, W1] 输入的特征图

        Returns
        -------
        _type_
            (B, C, H_out, W_out) 上采样后的特征图
        """
        
        grid = self.grid.repeat(input_tensor.size(
            0), 1, 1, 1).to(input_tensor.device)
        output_tensor = F.grid_sample(input_tensor, grid, align_corners=True)
        return output_tensor


if __name__ == '__main__':
    low = torch.randn([8, 128, 16, 16])
    high = torch.randn([8, 256, 8, 8])
    model = Semantic_flow(256, 128)
    model_x = model(high, low)
    print(model_x.size())

    x = torch.randn([1, 8, 64, 64])
    model_2 = Resampler(64, 256)
    model2_x = model_2(x)
    print(model2_x.size())
