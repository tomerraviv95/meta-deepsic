import torch as tc

class DeepSICNet(tc.nn.Module):
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

    def __init__(self, conf, input_size=1, batch_size=None):
        super(DeepSICNet, self).__init__()
        self.conf = conf
        self.numHiddenUnits = 60
        self.numClasses = int(len(self.conf.v_fConst))  # = 2 --> Binary Classification
        self.inputSize = input_size  # Batch_size: training or testing
        #         self.LSTM = tc.nn.LSTM(1, self.numHiddenUnits, batch_first=True) # Input (batch, seq_len=N+k-1, input_size=1)
        self.fullyConnectedLayer0 = tc.nn.Linear(self.conf.K + self.conf.N - 1, self.numHiddenUnits)
        #         self.h0 = tc.randn(1, batch_size, 60) # Initial hidden states for each element in the batch
        #         self.c0 = tc.randn(1, batch_size, 60) # Initial cell states for each element in the batch
        self.sigmoid = tc.nn.Sigmoid()
        self.fullyConnectedLayer1 = tc.nn.Linear(self.numHiddenUnits, int(self.numHiddenUnits / 2))
        self.reluLayer = tc.nn.ReLU()
        self.fullyConnectedLayer2 = tc.nn.Linear(int(self.numHiddenUnits / 2), self.numClasses)
        self.final_layer = tc.nn.Identity()

    def forward(self, m_fYtrain):
        out0 = self.sigmoid(self.fullyConnectedLayer0(m_fYtrain.squeeze(-1)))
        fc1_out = self.reluLayer(self.fullyConnectedLayer1(out0))
        out = self.final_layer(self.fullyConnectedLayer2(fc1_out))
        return out