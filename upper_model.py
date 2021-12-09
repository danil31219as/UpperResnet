from uper_head import UPerHead
import torch.nn as nn
from utils import resize

class UpperModel(nn.Module):
    def __init__(self, img_size=512, pretrained_path=''):
      super().__init__()
      self.backbone = None
      self.head = UPerHead(in_channels=[768, 768, 768, 768],
        num_classes=1,
        channels=768, in_index=[0, 1, 2, 3])
      self.align_corners = self.head.align_corners

    def forward(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.backbone(img)
        out = self.head(x)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out