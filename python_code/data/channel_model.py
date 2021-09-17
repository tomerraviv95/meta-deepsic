from python_code.utils.config_singleton import Config
from python_code.plotting.plotter_config import *
from dir_definitions import RESOURCES_DIR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

conf = Config()


class ChannelModel:
    @staticmethod
    def calculate_channel(n_ant, n_user, frame_num, iteration, phase) -> np.ndarray:
        pass

    @staticmethod
    def calculate_channel_wrapper(channel_mode, n_ant, n_user, phase, frame_num, iteration, fading) -> np.ndarray:
        if channel_mode == 'SED' or phase == 'train':
            H = SEDChannel.calculate_channel(n_ant, n_user, frame_num, iteration, phase)
        elif channel_mode == 'Gaussian':
            H = GaussianChannel.calculate_channel(n_ant, n_user, frame_num, iteration, phase)
        elif channel_mode == 'COST' and not fading:
            H = COSTChannel.calculate_channel(n_ant, n_user, frame_num, iteration, phase)
        else:
            raise NotImplementedError
        return H

    @staticmethod
    def get_channel(channel_mode, n_ant, n_user, phase, fading, frame_num, iteration):
        H = ChannelModel.calculate_channel_wrapper(channel_mode, n_ant, n_user, phase, frame_num, iteration, fading)
        H = ChannelModel.add_fading(H, fading, phase, n_ant, iteration)
        return H

    @staticmethod
    def add_fading(H, fading, phase, n_ant, iteration):
        if fading:

            if conf.fading_type == 1:
                if phase == 'train':
                    degs_array = np.array([1, 1, 1, 1])
                    center = 1
                else:
                    degs_array = np.array([51, 39, 33, 21])
                    center = 0.8
            elif conf.fading_type == 2:
                if phase == 'train':
                    degs_array = np.array([51, 39, 33, 21])
                    center = 0.8
                else:
                    degs_array = np.array([28, 31, 23, 18])
                    center = 0.6
            elif conf.fading_type == 3:
                if phase == 'train':
                    degs_array = np.array([51, 39, 33, 21])
                    center = 0.9
                else:
                    degs_array = np.array([51, 39, 33, 21])
                    center = 0.7
            else:
                raise ValueError("No such fading type!!!")

            fade_mat = center + (1 - center) * np.cos(2 * np.pi * iteration / degs_array)
            if conf.change_user_only:
                remaining_indices = list(set(list(range(n_ant))) - set([conf.change_user_only]))
                fade_mat[remaining_indices] = 1
            fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
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
        print(iteration)
        main_folder = 1 + (iteration // 10)
        for i in range(1, n_user + 1):
            path_to_mat = os.path.join(RESOURCES_DIR, f'{phase}_{main_folder}', f'h_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['norm_channel'][iteration % 10]
            total_h[i - 1] = 0.25 * h_user
        total_h[np.arange(n_user), np.arange(n_user)] = 1
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
        print(h)
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

        if i == 2:
            break
