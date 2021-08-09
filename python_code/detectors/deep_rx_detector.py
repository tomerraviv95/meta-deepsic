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

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(in_features=s_nK+s_nN-1, out_features=60, bias=True)
      (sigmoid): Sigmoid()
      (fullyConnectedLayer): Linear(in_features=60, out_features=30, bias=True)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(in_features=30, out_features=2, bias=True)
    ================================
    Note:
    The output of the network is not probabilities,
    to obtain probabilities apply a softmax function to the output, viz.
    output = DeepSICNet(data)
    probs = torch.softmax(output, dim), for a batch inference, set dim=1; otherwise dim=0.
    """

    def __init__(self):
        super(DeepRXDetector, self).__init__()
        self.main = nn.Sequential(
            [ResnetBlock() for _ in range(RESNET_BLOCKS)]
        )
        self.fc0 = nn.Linear(conf.n_user + conf.n_ant - 1, HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE / 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(HIDDEN_SIZE / 2), CLASSES_NUM)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out0 = self.sigmoid(self.fc0(y.squeeze(-1)))
        fc1_out = self.relu(self.fc1(out0))
        out = self.fc2(fc1_out)
        return out
