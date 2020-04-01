import torch
import torch.nn.functional as F
from torch import nn


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


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


class DANet(nn.Module):
    def __init__(self, alpha):
        super(DANet, self).__init__()
        self.conv_layer1 = self._make_conv_layer(out_channels=16, k_size=4)
        self.conv_layer2 = self._make_conv_layer(in_channels=16, out_channels=24)
        self.conv_layer3 = self._make_conv_layer(in_channels=24, out_channels=28)
        self.conv_layer4 = self._make_conv_layer(in_channels=28, out_channels=34)
        self.conv_layer5 = self._make_conv_layer(in_channels=34, out_channels=42)
        self.conv_layer6 = self._make_conv_layer(in_channels=42, out_channels=50)
        self.conv_layer7 = self._make_conv_layer(in_channels=50, out_channels=50)
        self.final_layer = self._make_conv_layer(in_channels=50, out_channels=1, activation=False)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3_500, 1_000)
        self.linear2 = nn.Linear(1_000, 256)
        self.linear3 = nn.Linear(256, 6)
        self.alpha = alpha

    def forward(self, x):

        x = self.conv_layer1(x)
        # Max pooling over a (2, 2, 2) window
        x = F.max_pool3d(x, (2, 2, 2))
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)

        categorical_branch = F.max_pool3d(x, (4, 4, 4))
        categorical_branch = GradientReversalLayer.apply(categorical_branch, self.alpha)
        categorical_branch = self.flatten(categorical_branch)

        categorical_branch = self.linear1(categorical_branch)
        categorical_branch = torch.relu(categorical_branch)

        categorical_branch = self.linear2(categorical_branch)
        categorical_branch = torch.relu(categorical_branch)

        categorical_branch = self.linear3(categorical_branch)

        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)

        return x, categorical_branch

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
