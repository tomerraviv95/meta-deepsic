from python_code.ecc.wrappers import encoder
from python_code.utils.constants import Phase
from python_code.data.channel_model import ChannelModel
from python_code.utils.config_singleton import Config
from torch.utils.data import Dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


def bpsk_modulate(b: np.ndarray) -> np.ndarray:
    return (-1) ** b


def calculate_sigma_from_snr(snr: int) -> float:
    """
    converts the Desired SNR into the noise power (noise variance)
    :param snr: signal-to-noise ratio
    :return: noise's sigma
    """
    return 10 ** (-0.1 * snr)


class DataGenerator(Dataset):
    """
    The Data Generator Class
    """

    def __init__(self, frame_size: int, phase: Phase, frame_num: int):
        super(DataGenerator).__init__()
        self.frame_size = frame_size
        self.phase = phase
        self.frame_num = frame_num
        self.channel_model = ChannelModel(self.phase)

    def __call__(self, snr: int):
        """
        Generates the input-channel symbols and output-channel values
        :param snr: signal-to-noise ratio
        :return: information word and received channel values
        """
        b_total = torch.empty(0)
        y_total = torch.empty(0)

        for frame_ind in range(self.frame_num):
            # get channel
            H = self.channel_model.get_channel(frame_ind)
            # generate bits
            b = np.random.randint(0, 2, size=(self.frame_size, conf.n_user))
            # encoding
            c = encoder(b, self.phase)
            # modulation
            x = bpsk_modulate(c)
            # pass through channel
            sigma = calculate_sigma_from_snr(snr)
            y = np.matmul(x, H) + np.sqrt(sigma) * np.random.randn(x.shape[0], conf.n_ant)
            if not conf.linear_channel:
                y = np.tanh(0.5 * y)
            # add to buffer
            b_total = torch.cat([b_total, torch.FloatTensor(b)])
            y_total = torch.cat([y_total, torch.FloatTensor(y)])

        return b_total.to(device), y_total.to(device)
