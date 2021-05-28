from python_code.utils.config_singleton import Config
from python_code.plotting.plotter_config import *
from dir_definitions import RESOURCES_DIR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

COST_CHANNELS = 5
MAX_FRAMES = 50
# MIN_COST_VAL = -0.0011940332244104606 # slow
MIN_COST_VAL = -0.002187582111148217  # fast

# MAX_COST_VAL = 0.0016165160263559176 # slow
MAX_COST_VAL = 0.002073063725906618  # fast

conf = Config()

class ChannelModel:
    @staticmethod
    def calculate_channel(n_ant, n_user, frame_num, iteration, phase) -> np.ndarray:
        pass

    @staticmethod
    def calculate_channel_wrapper(channel_mode, n_ant, n_user, phase, frame_num, iteration) -> np.ndarray:
        if channel_mode == 'SED':
            H = SEDChannel.calculate_channel(n_ant, n_user, frame_num, iteration, phase)
        elif channel_mode == 'Gaussian':
            H = GaussianChannel.calculate_channel(n_ant, n_user, frame_num, iteration, phase)
        elif channel_mode == 'COST':
            H = COSTChannel.calculate_channel(n_ant, n_user, frame_num, iteration, phase)
        else:
            raise NotImplementedError
        return H

    @staticmethod
    def get_channel(channel_mode, n_ant, n_user, phase, fading, frame_num, iteration):
        H = ChannelModel.calculate_channel_wrapper(channel_mode, n_ant, n_user, phase, frame_num, iteration)
        H = ChannelModel.add_fading(H, fading, phase, n_user, iteration)
        return H

    @staticmethod
    def add_fading(H, fading, phase, n_user, iteration):
        if phase == 'test' and fading:
            degs_array = np.array([51, 39, 33, 21])
            fade_mat = (0.8 + 0.2 * np.cos(2 * np.pi * iteration / degs_array))
            if conf.change_user_only:
                remaining_indices = list(set(list(range(n_user))) - set([conf.change_user_only]))
                fade_mat[remaining_indices] = 1
            fade_mat = np.tile(fade_mat.reshape(1, -1), [n_user, 1]).T
            H = H * fade_mat
        return H


class SEDChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user, frame_num, iteration, phase) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        return H


class GaussianChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user, frame_num, iteration, phase) -> np.ndarray:
        return np.random.randn(n_ant, n_user)


class COSTChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user, frame_num, iteration, phase) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        if phase == 'train':
            phase_shift = 0
        else:
            phase_shift = frame_num

        for i in range(1, n_user // 2 + 1):
            path_to_mat = os.path.join(RESOURCES_DIR, phase,
                                       f'h_link_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['h1'][iteration + phase_shift]
            h_users1 = np.concatenate([np.real(h_user), np.imag(h_user)], axis=1)
            h_users2 = np.concatenate([-np.imag(h_user), np.real(h_user)], axis=1)
            h_users = np.concatenate([h_users1, h_users2], axis=0)
            row_ind = (i - 1) % 2
            column_ind = (i - 1) // 2
            total_h[row_ind * n_ant // 2:(row_ind + 1) * n_ant // 2,
            column_ind * n_ant // 2:(column_ind + 1) * n_ant // 2] = h_users

        total_h = (total_h - MIN_COST_VAL) / (MAX_COST_VAL - MIN_COST_VAL)
        return total_h


if __name__ == "__main__":
    channel_model = ChannelModel()

    total_h_train = np.empty([conf.n_ant, conf.n_user, 0])
    for iteration in range(conf.train_frame_num):
        h = channel_model.get_channel(conf.channel_mode, conf.n_ant, conf.n_user,
                                      'train', conf.fading, conf.train_frame_num, iteration)
        total_h_train = np.concatenate([total_h_train, np.expand_dims(h, axis=2)], axis=2)

    total_h_test = np.empty([conf.n_ant, conf.n_user, 0])
    for iteration in range(conf.test_frame_num):
        h = channel_model.get_channel(conf.channel_mode, conf.n_ant, conf.n_user,
                                      'test', conf.fading, conf.test_frame_num, iteration)
        total_h_test = np.concatenate([total_h_test, np.expand_dims(h, axis=2)], axis=2)

    for i in range(conf.n_ant):

        plt.figure()
        for j in range(conf.n_user):
            plt.plot(total_h_train[i, j], label=f'{j}')
        plt.ylabel(r'magnitude', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        # plt.title('Train Channel')
        plt.ylim([-0.1, 1.1])
        plt.grid(True, which='both')
        plt.legend(loc='upper left', prop={'size': 15})
        plt.show()

        plt.figure()
        for j in range(conf.n_user):
            plt.plot(total_h_test[i, j], label=f'{j}')
        plt.ylabel(r'magnitude', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        # plt.title('Test Channel')
        plt.ylim([-0.1, 1.1])
        plt.grid(True, which='both')
        plt.legend(loc='upper left', prop={'size': 15})
        plt.show()
