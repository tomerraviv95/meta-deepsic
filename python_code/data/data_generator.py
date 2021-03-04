from python_code.data.channel_model import ChannelModel
from python_code.utils.config_singleton import Config
from python_code.utils.utils import calculate_sigma_from_snr
from torch.utils.data import Dataset
import concurrent.futures
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


class DataGenerator(Dataset):
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

    def __init__(self, size):
        super(DataGenerator).__init__()
        self.size = size
        self.H = ChannelModel.get_channel(conf.ChannelModel, conf.N, conf.K, conf.csi_noise)

    def generate_symbols(self):
        # generate bits
        b = np.random.randint(0, 2, size=(self.size, conf.K))
        # generate symbols
        x = (-1) ** b
        # return symbols tensor
        return torch.FloatTensor(x).unsqueeze(-1)

    def __call__(self, snr):
        x = self.generate_symbols()
        sigma = calculate_sigma_from_snr(snr)
        y = torch.matmul(self.H, x) + torch.sqrt(sigma) * torch.randn(self.size, conf.N, 1)
        return x.to(device), y.to(device)
