import torch.nn as nn


class Semantic_Segmentater(nn.Module):
    def __init__(self, channels=3, class_num=40):
        super(Semantic_Segmentater, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=10, kernel_size=9, stride=1, padding=4,
                      padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=9, stride=1, padding=4,
                      padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=20, out_channels=class_num, kernel_size=9, stride=1, padding=4,
                      padding_mode='reflect'),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model(x)
        return x
