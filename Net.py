import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = self._make_conv_layer(out_channels=16, k_size=4)
        self.conv_layer2 = self._make_conv_layer(in_channels=16, out_channels=24)
        self.conv_layer3 = self._make_conv_layer(in_channels=24, out_channels=28)
        self.conv_layer4 = self._make_conv_layer(in_channels=28, out_channels=34)
        self.conv_layer5 = self._make_conv_layer(in_channels=34, out_channels=42)
        self.conv_layer6 = self._make_conv_layer(in_channels=42, out_channels=50)
        self.conv_layer7 = self._make_conv_layer(in_channels=50, out_channels=50)
        self.final_layer = self._make_conv_layer(in_channels=50, out_channels=1, activation=False)

    def forward(self, x):

        x = self.conv_layer1(x)
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(x, (2, 2, 2))
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)

        return x

    @staticmethod
    def _make_conv_layer(out_channels, in_channels=1, k_size=5, activation=True):
        if activation:
            conv_layer = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k_size, k_size, k_size),
                          padding=0),
                nn.LeakyReLU(),  # (!)Think of padding(!)
            )
        else:
            conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(k_size, k_size, k_size), padding=2)

        return conv_layer
