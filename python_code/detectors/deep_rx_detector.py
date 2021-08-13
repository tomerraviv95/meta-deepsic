from python_code.utils.config_singleton import Config
from torch.nn import functional as F
from torch import nn
import torch

conf = Config()

CLASSES_NUM = 2
HIDDEN_SIZE = 60
RESNET_BLOCKS = 11


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResnetBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


class DeepRXDetector(nn.Module):
    """
    The DeepRXDetector Network Architecture
    """

    def __init__(self):
        super(DeepRXDetector, self).__init__()
        self.all_blocks = nn.Sequential(
            [ResnetBlock(conf.n_ant, CLASSES_NUM) for _ in range(RESNET_BLOCKS)]
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.all_blocks(y)
