from torch import nn

CLASSES_NUM = 2


class DeepSICNet(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(in_features=s_nK+s_nN-1, out_features=60, bias=True)
      (sigmoid): Sigmoid()
      (fullyConnectedLayer): Linear(in_features=60, out_features=30, bias=True)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(in_features=30, out_features=2, bias=True)
      (final_layer): Identity()
    ================================
    Note:
    The output of the network is not probabilities,
    to obtain probabilities apply a softmax function to the output, viz.
    output = DeepSICNet(data)
    probs = torch.softmax(output, dim), for a batch inference, set dim=1; otherwise dim=0.
    """

    def __init__(self, conf, input_size=1):
        super(DeepSICNet, self).__init__()
        self.conf = conf
        self.hidden_size = 60
        self.input_size = input_size  # Batch_size: training or testing
        self.fc0 = nn.Linear(self.conf.K + self.conf.N - 1, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(self.hidden_size, int(self.hidden_size / 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(self.hidden_size / 2), CLASSES_NUM)
        self.identity = nn.Identity()

    def forward(self, y):
        out0 = self.sigmoid(self.fc0(y.squeeze(-1)))
        fc1_out = self.relu(self.fc1(out0))
        out = self.identity(self.fc2(fc1_out))
        return out
