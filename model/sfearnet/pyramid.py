import torch
import torch.nn as nn
from thop import profile
from torch.nn import functional as F
from .CBAM import CBAM


class Pyramid_Extraction(nn.Module):
    def __init__(self, channel, rate=1, bn_mom=0.1):
        super(Pyramid_Extraction, self).__init__()
        self.channel = channel

        self.branch1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=rate, dilation=rate, groups=channel,
                      bias=True),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1),
                      stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(
                1, 1), padding=1, groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1),
                      stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=4 * rate, dilation=4 * rate,
                      groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1),
                      stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=(1, 1), padding=8 * rate, dilation=8 * rate,
                      groups=channel, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=(1, 1),
                      stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5_conv = nn.Conv2d(channel, channel, kernel_size=(
            1, 1), stride=(1, 1), padding=0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(channel, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d((channel) * 5, (channel) * 5, kernel_size=(1, 1), stride=(1, 1), padding=0,
                      groups=(channel) * 5, bias=False),
            nn.BatchNorm2d((channel) * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d((channel) * 5, channel, kernel_size=(1, 1),
                      stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1_1 = self.branch1(x)
        conv3_1 = self.branch2(x)
        conv3_2 = self.branch3(x)
        conv3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(
            global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat(
            [conv1_1, conv3_1, conv3_2, conv3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class Pyramid_Merge(nn.Module):
    def __init__(self, channel, diff_mode: str = "abs"):
        """_summary_
        TODO: 1.把两个参数描述写成中文版。2.将上边的summery替换成作用（不是指的逻辑）
        Args:
            channel (_type_): _description_
            diff_mode (str, optional): _description_. Defaults to "abs".
        """
        super(Pyramid_Merge, self).__init__()
        self.channel = channel
        self.diff_mode = diff_mode
        self.cbam = CBAM(channel)
        self.pe = Pyramid_Extraction(channel)

        self.conv = nn.Sequential(
            nn.Conv2d(channel*2, channel*2, kernel_size=1, groups=channel*2),
            nn.BatchNorm2d(channel*2),
            nn.ReLU(),
            nn.Conv2d(channel*2, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def _compute_difference(self, feat_b: torch.Tensor, feat_a: torch.Tensor) -> torch.Tensor:
        """计算双时相特征值的差异（已完成）
        TODO: 1.把两个参数描述写成中文版。2.将上边的summery替换成作用（不是指的逻辑）
        Args:
            feat_b (torch.Tensor): [B, C, H, W] - features from second temporal phase
            feat_a (torch.Tensor): [B, C, H, W] - features from first temporal phase

        Returns:
            torch.Tensor: [B, C, H, W] - difference between the two feature maps
        """
        abs_diff = torch.abs(feat_b - feat_a)
        if self.diff_mode == "abs":
            return abs_diff
        elif self.diff_mode == "DHDM":
            #TODO:给这个分支写注释，解释DHDM怎么算的差距。要用概括的语言描述，不要用代码的方式描述。要解释为什么要用这个方法来计算差距。
            
            # 原始版本
            cos_sim = F.cosine_similarity(feat_b, feat_a, dim=1, eps=1e-6).unsqueeze(1)
            direction_term = 1.0 - cos_sim
            return abs_diff * direction_term
        elif self.diff_mode == "DHDM05":
            # 标注一下，这个版本是后边跑消融实验codex帮我改的版本
            # [B, 1, H, W],将所有比1e-6的值全部强制改成1e-6，防止除0错误，限制最小值。另外.norm是得到标量，每个向量的长度
            feat_b_norm = feat_b.norm(dim=1, keepdim=True).clamp_min(1e-6)
            feat_a_norm = feat_a.norm(dim=1, keepdim=True).clamp_min(
                1e-6)  # [B, 1, H, W],将所有比1e-6的值全部强制改成1e-6，防止除0错误，限制最小值。
            #  先逐元素相乘，再沿着通道维度求和，得到[B,1,H,W]，表示两个特征图的余弦相似度
            cos_sim = (feat_b * feat_a).sum(dim=1, keepdim=True) / \
                (feat_b_norm * feat_a_norm)
            #如果是没有系数0.5，当前版本更激进：权重范围 [0, 2]。0.5版本更温和：权重范围 [0.5, 1.5]
            direction_weight = 1.0 - 0.5 * cos_sim #TODO: 1.为什么要用1-cos_sim。2.为什么要给cos_sim乘以0.5，为什么不是1.0。3.为什么要用1-cos_sim*0.5，而不是直接用cos_sim*0.5
            return abs_diff * direction_weight
        else:
            raise ValueError(
                f"Invalid diff_mode: {self.diff_mode}. Supported modes are 'abs' ,'DHDM05' and 'DHDM'.")

    def forward(self, feat_b, feat_a):
        """_summary_
        TODO: 1.把两个参数描述写成中文版。2.将上边的summery替换成作用（不是指的逻辑）
        Args:
            feat_b (_type_): _description_
            feat_a (_type_): _description_

        Returns:
            _type_: _description_
        """
        input_cat = torch.cat([feat_b, feat_a], dim=1)
        input_abs = self._compute_difference(feat_b, feat_a)

        input_cat_conv = self.conv(input_cat)
        input_cat_conv_cbam = self.cbam(input_cat_conv)

        input_abs_py = self.pe(input_abs)
        input_abs_py_abs = input_abs_py+input_abs

        out1 = input_abs_py_abs+input_cat_conv_cbam
        return out1


if __name__ == "__main__":
    model = Pyramid_Merge(256)
    x1 = torch.randn([8, 256, 64, 64])
    x2 = torch.randn([8, 256, 64, 64])
    model_x = model(x1, x2)
    print(model_x.size())

    input_data1 = torch.randn([1, 256, 64, 64])
    input_data2 = torch.randn([1, 256, 64, 64])
    flops, params = profile(model, inputs=(input_data1, input_data2,))
    print('Number of parameters: ' + str(params))
    print('FLOPs: ' + str(flops))
    print(params / 10 ** 6)
    print(flops / 10 ** 9)
