from python_code.data.channels.sed_channel import SEDChannel
from python_code.data.channels.cost_channel import COSTChannel
from python_code.utils.config_singleton import Config
from python_code.plotting.plotter_config import *
from python_code.utils.constants import Phase
import matplotlib.pyplot as plt
from enum import Enum, auto
import numpy as np

conf = Config()


class Channel(Enum):
    SED = auto()
    COST = auto()


class ChannelModel:
    def __init__(self, phase: Phase):
        self.phase = phase

    def get_channel(self, frame_ind: int) -> np.ndarray:
        # create SED channel matrix
        if conf.channel_mode == Channel.SED.name or self.phase == Phase.TRAIN:
            H = SEDChannel.calculate_channel(conf.n_ant, conf.n_user, self.phase, frame_ind, conf.fading,
                                             conf.change_user_only)
        # create COST channel matrix
        elif conf.channel_mode == Channel.COST.name:
            H = COSTChannel.calculate_channel(conf.n_ant, conf.n_user, frame_ind, self.phase, conf.change_user_only)
        else:
            raise NotImplementedError
        return H


def plot_channel_by_phase(phase: Phase):
    """
    Plot for one given antenna at a time
    :param phase: Enum
    """
    channel_model = ChannelModel(phase)

    total_h = np.empty([conf.n_ant, conf.n_user, 0])
    for frame_ind in range(conf.train_frame_num):
        h = channel_model.get_channel(frame_ind)
        total_h = np.concatenate([total_h, np.expand_dims(h, axis=2)], axis=2)

    for i in range(conf.n_ant):
        plt.figure()
        for j in range(conf.n_user):
            plt.plot(total_h[i, j], label=f'{j}')
        plt.ylabel(r'magnitude', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        # plt.title('Train Channel')
        plt.ylim([-0.1, 1.1])
        plt.grid(True, which='both')
        plt.legend(loc='upper left', prop={'size': 15})
        plt.show()


if __name__ == "__main__":
    phase = Phase.TEST  # either TRAIN or TEST
    plot_channel_by_phase(phase)
