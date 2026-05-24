import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        # # Layer 1: Two conv layers with BatchNorm
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        # # Layer 2: Distance-based loss with margin
        self.margin = margin
        self.max_loss = max(1.0, 0.5 * margin * margin)

    def forward(self, feat1, feat2, label):
        """
        feat1, feat2: [B, C, H, W] - features from two temporal phases
        label: [B, H, W] - change label, 0 for no change, 1 for change
        SFEARNet input shapes: input1=(8, 3, 256, 256), input2=(8, 3, 256, 256)
        SFEARNet output shapes: x=(8, 2, 256, 256), edge=(8, 2, 256, 256), feat1=(8, 256, 256, 256), feat2=(8, 256, 256, 256)
        ContrastiveLoss inputs: feat1=(8, 256, 256, 256), feat2=(8, 256, 256, 256), label=(8, 256, 256)
        """
        # Print input shapes once for debugging
        if not hasattr(self, '_printed_input_shapes'):
            print(f"ContrastiveLoss inputs: feat1={tuple(feat1.shape)}, feat2={tuple(feat2.shape)}, label={tuple(label.shape)}")
            self._printed_input_shapes = True

        # Process features through conv layers
        # feat1 = F.relu(self.bn1(self.conv1(feat1)))
        # feat2 = F.relu(self.bn1(self.conv1(feat2)))
        # feat1 = F.relu(self.bn2(self.conv2(feat1)))
        # feat2 = F.relu(self.bn2(self.conv2(feat2)))

        # L2 normalize along channel dimension
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)

        # Compute Euclidean distance for each pixel
        dist = torch.sqrt(torch.sum((feat1_norm - feat2_norm)**2, dim=1))  # [B, H, W]

        # Contrastive loss: (1-label) * 0.5 * dist^2 + label * 0.5 * max(0, margin - dist)^2
        #经典 contrastive loss 的形式：相似样本拉近，不相似样本推远，并且距离大于 margin 时不再继续惩罚。
        loss = (1 - label) * 0.5 * dist**2 + label * 0.5 * torch.clamp(self.margin - dist, min=0)**2
        #归一化
        loss = loss.mean() / self.max_loss
        #保证在[0,1]之间
        return torch.clamp(loss, 0.0, 1.0)