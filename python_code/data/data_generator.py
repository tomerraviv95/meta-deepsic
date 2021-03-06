from python_code.data.channel_model import ChannelModel
from python_code.ecc.rs_main import encode
from python_code.utils.config_singleton import Config
from python_code.utils.utils import calculate_sigma_from_snr
from torch.utils.data import Dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


def bpsk_modulate(b):
    """
    Generates a Tensor of Uniformly Distributed BPSK Symbols
    :return: a Tensor of Size: [Batch_Size, <# of RX_Antennas>, <# of Users>]
    """
    # generate symbols
    return (-1) ** b


class DataGenerator(Dataset):
    """
    The Data Generator Class
    """

    def __init__(self, frame_size, phase):
        super(DataGenerator).__init__()
        self.frame_size = frame_size
        self.phase = phase

    def __call__(self, snr):
        """
        Generates the input-channel symbols and output-channel values
        :param snr: signal-to-noise ratio
        :return: symbols and received channel values
        """
        b_total = torch.empty(0)
        x_total = torch.empty(0)
        y_total = torch.empty(0)

        if conf.use_ecc and self.phase == 'test':
            encoding = lambda b: encode(b, conf.n_ecc_symbols)
        else:
            encoding = lambda b: b

        for i in range(conf.frame_num):
            # get channel
            H = ChannelModel.get_channel(conf.ChannelModel, conf.n_ant, conf.n_user, conf.csi_noise, self.phase)
            # generate bits
            b = np.random.randint(0, 2, size=(self.frame_size, conf.n_user))
            c = np.concatenate([encoding(b[:, i]).reshape(-1, 1) for i in range(b.shape[1])], axis=1)
            x = bpsk_modulate(c)
            sigma = calculate_sigma_from_snr(snr)
            y = np.matmul(x, H) + np.sqrt(sigma) * np.random.randn(x.shape[0], conf.n_ant)
            b_total = torch.cat([b_total, torch.FloatTensor(b)])
            x_total = torch.cat([x_total, torch.FloatTensor(x)])
            y_total = torch.cat([y_total, torch.FloatTensor(y)])
        return b_total.unsqueeze(-1).to(device), x_total.unsqueeze(-1).to(device), y_total.unsqueeze(-1).to(device)
