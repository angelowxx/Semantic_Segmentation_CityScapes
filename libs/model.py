import torch.nn as nn


class Semantic_Segmentater(nn.Module):
    def __init__(self, channels=3, class_num=40):
        super(Semantic_Segmentater, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.MaxPool2d(3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=class_num, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.MaxPool2d(3, stride=2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=class_num, out_channels=class_num, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect'),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=class_num, out_channels=class_num, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect'),
            nn.Softmax(dim=-3),
        )

    def forward(self, x):
        x = self.model(x)
        return x
