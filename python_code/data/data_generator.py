from python_code.utils.utils import Utils
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataGenerator(Utils):
    """
    The Data Generator Class

    Attributes
    ----------
    conf : a Conf class instance

    Methods
    -------
    getH(channel_mode: str)
        Sample the channel matrix from the specified model in the Conf class
        Supporting Channel Models: Spatial Exponential Decay (SED)    --> H_{i,j} = exp(-|i-j|)
                                   Gaussian Channel Matrix (Gaussian) --> H_{i,j} ~ N(0,1)
    getSymbols(batch_size: int)
        Generates a Tensor of Uniformly Distributed BPSK Symbols
        Returns a Tensor of Size: [Batch_Size, <# of RX_Antennas>, <# of Users>]

    __call__(snr: float)
        Generates a Data Dictionary Containing data['key':value] as Follows:
        ['m_fStrain']    -> Training Symbols (Labels) - Size [Training_Batch_size, <# of Users>]
        ['m_fSTest']     -> Testing Symbols (Labels)  - Size [Test_Batch_size, <# of Users>]
        ['m_fYtest']     -> Output of the channel for the Testing Symbols ['m_fSTest']
        ['m_fYtrain']    -> Output of the Channel for the Training Symbols ['m_fStrain']
        ['m_fRtrain']    -> Received Signals from a Corrupted Channel without noise
        ['m_fYtrainErr'] -> Output of the Corrupted Channel + Noise: ['m_fRtrain'] + Noise

    """

    def __init__(self, config):
        super(DataGenerator).__init__()
        self.conf = config
        self.H = self.getH(channel_mode=self.conf.ChannelModel)

    def getH(self, channel_mode):
        if channel_mode == 'SED':
            H_row = torch.FloatTensor([i for i in range(self.conf.N)])
            H_row = H_row.repeat([self.conf.K, 1]).t()
            H_column = torch.FloatTensor([i for i in range(self.conf.K)])
            H_column = H_column.repeat([self.conf.N, 1])
            H = torch.exp(-torch.abs(H_row - H_column))
        elif channel_mode == 'Gaussian':
            H = torch.randn(self.conf.N, self.conf.K)
        else:
            raise NotImplementedError
        return H

    def getSymbols(self, batch_size):
        return torch.FloatTensor([[np.random.choice(self.conf.v_fConst) \
                                   for _ in range(self.conf.K)] \
                                  for _ in range(batch_size)]).unsqueeze(-1)

    def __call__(self, snr=0.):
        m_fStrain = self.getSymbols(self.conf.train_size)
        m_fStest = self.getSymbols(self.conf.test_size)
        # Generating noisy CSI
        m_fRtrain = torch.zeros(self.conf.train_size, self.conf.N, 1)
        for i in range(self.conf.frame_num):
            frame_idxs = torch.arange(i * self.conf.frame_size, (i + 1) * self.conf.frame_size)
            curr_H_noise = (1. + torch.sqrt(torch.FloatTensor([self.conf.std]))) * torch.randn(self.H.shape)
            curr_Hn = torch.mul(self.H, curr_H_noise)
            curr_x = m_fStrain[frame_idxs, :]
            m_fRtrain[frame_idxs, :] = torch.matmul(curr_Hn, curr_x)
        s_fSigW = self.fSNRToW(snr)
        m_fYtrain = torch.matmul(self.H, m_fStrain) \
                    + torch.sqrt(s_fSigW) * torch.randn(self.conf.train_size, self.conf.N, 1)
        m_fYtrainErr = m_fRtrain + torch.sqrt(s_fSigW) * torch.randn(self.conf.train_size, self.conf.N, 1)
        m_fYtest = torch.matmul(self.H, m_fStest) + torch.sqrt(s_fSigW) * torch.randn(self.conf.test_size,
                                                                                      self.conf.N, 1)
        self.data = {'m_fStrain': m_fStrain.to(device), 'm_fStest': m_fStest.to(device),
                     'm_fRtrain': m_fRtrain.to(device),
                     'm_fYtrain': m_fYtrain.to(device), 'm_fYtrainErr': m_fYtrainErr.to(device),
                     'm_fYtest': m_fYtest.to(device)}
        return self.data
