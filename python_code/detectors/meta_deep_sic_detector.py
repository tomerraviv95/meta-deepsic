from python_code.utils.config_singleton import Config
from torch.nn import functional as F
from torch import nn
import torch

conf = Config()

CLASSES_NUM = 2
HIDDEN_SIZE = 60


class MetaDeepSICDetector(nn.Module):
    """
    The Meta DeepSIC Network Architecture

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
        super(MetaDeepSICDetector, self).__init__()
        self.fc0 = nn.Linear(conf.n_user + conf.n_ant - 1, HIDDEN_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE / 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(HIDDEN_SIZE / 2), CLASSES_NUM)

    def forward(self, y: torch.Tensor, var: list) -> torch.Tensor:
        fc_out0 = F.linear(y.squeeze(-1), var[0], var[1])
        out0 = torch.sigmoid(fc_out0)
        fc_out1 = F.linear(out0, var[2], var[3])
        out1 = nn.functional.relu(fc_out1)
        fc_out2 = F.linear(out1, var[4], var[5])
        return fc_out2
