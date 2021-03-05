from python_code.data.channel_model import ChannelModel
from python_code.utils.config_singleton import Config
from python_code.utils.utils import calculate_sigma_from_snr
from torch.utils.data import Dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = Config()


def generate_symbols():
    """
    Generates a Tensor of Uniformly Distributed BPSK Symbols
    :return: a Tensor of Size: [Batch_Size, <# of RX_Antennas>, <# of Users>]
    """
    # generate bits
    b = np.random.randint(0, 2, size=(conf.frame_size, conf.n_user))
    # generate symbols
    x = (-1) ** b
    # return symbols tensor
    return torch.FloatTensor(x).unsqueeze(-1)


class DataGenerator(Dataset):
    """
    The Data Generator Class
    """

    def __init__(self, size, phase):
        super(DataGenerator).__init__()
        self.size = size
        self.phase = phase

    def __call__(self, snr):
        """
        Generates the input-channel symbols and output-channel values
        :param snr: signal-to-noise ratio
        :return: symbols and received channel values
        """
        x_total = torch.empty(0)
        y_total = torch.empty(0)
        frame_num = int(self.size / conf.frame_size)
        if frame_num == 0:
            raise ValueError("Frame number is zero!!!")

        for i in range(frame_num):
            H = ChannelModel.get_channel(conf.ChannelModel, conf.n_ant, conf.n_user, conf.csi_noise, self.phase)
            x = generate_symbols()
            sigma = calculate_sigma_from_snr(snr)
            y = torch.matmul(H, x) + torch.sqrt(sigma) * torch.randn(conf.frame_size, conf.n_ant, 1)
            x_total = torch.cat([x_total, x])
            y_total = torch.cat([y_total, y])
        return x_total.to(device), y_total.to(device)
