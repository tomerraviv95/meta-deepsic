from python_code.data.channel_model import ChannelModel
from python_code.utils.config_singleton import Config
from python_code.utils.utils import db_to_scalar
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

    def __init__(self):
        super(DataGenerator).__init__()

    def generate_symbols(self, batch_size):
        # generate bits
        b = np.random.randint(0, 2, size=(batch_size, conf.K))
        # generate symbols
        x = (-1) ** b
        # return symbols tensor
        return torch.FloatTensor(x).unsqueeze(-1)

    def __getitem__(self, snr):
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.__call__, snr, database) for snr in [snr]]
        b, y = (np.concatenate(arrays) for arrays in zip(*database))
        b, y = torch.Tensor(b).to(device=device), torch.Tensor(y).to(device=device)
        return b, y

    def __call__(self, snr):
        H = ChannelModel.get_channel(conf.ChannelModel, conf.N, conf.K)
        m_fStrain = self.generate_symbols(conf.train_size)
        m_fStest = self.generate_symbols(conf.test_size)
        s_fSigW = db_to_scalar(snr)
        m_fYtrain = torch.matmul(H, m_fStrain) + torch.sqrt(s_fSigW) * torch.randn(conf.train_size, conf.N, 1)
        m_fYtest = torch.matmul(H, m_fStest) + torch.sqrt(s_fSigW) * torch.randn(conf.test_size, conf.N, 1)
        self.data = {'m_fStrain': m_fStrain.to(device), 'm_fStest': m_fStest.to(device),
                     'm_fYtrain': m_fYtrain.to(device), 'm_fYtest': m_fYtest.to(device)}
        return self.data
