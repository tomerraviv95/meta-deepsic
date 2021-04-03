from python_code.utils.config_singleton import Config
from dir_definitions import RESOURCES_DIR
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os


class ChannelModel:
    @staticmethod
    def calculate_channel(n_ant, n_user, iteration) -> np.ndarray:
        pass

    @staticmethod
    def calculate_channel_wrapper(channel_mode, n_ant, n_user, phase, iteration) -> np.ndarray:
        if channel_mode == 'SED':
            H = SEDChannel.calculate_channel(n_ant, n_user, iteration)
        elif channel_mode == 'Gaussian':
            H = GaussianChannel.calculate_channel(n_ant, n_user, iteration)
        # elif channel_mode == 'COST' and phase == 'train':
        #     H = SEDChannel.calculate_channel(n_ant, n_user, iteration)
        elif channel_mode == 'COST':  # and phase == 'test'
            H = COSTChannel.calculate_channel(n_ant, n_user, iteration)
        else:
            raise NotImplementedError
        return H

    @staticmethod
    def get_channel(channel_mode, n_ant, n_user, csi_noise, phase, fading, iteration):
        H = ChannelModel.calculate_channel_wrapper(channel_mode, n_ant, n_user, phase, iteration)
        H = ChannelModel.noising_channel(H, csi_noise, phase)
        H = ChannelModel.add_fading(H, fading, phase, n_user, iteration)
        return H

    @staticmethod
    def noising_channel(H, csi_noise, phase):
        if phase == 'test' and csi_noise > 0:
            curr_H_noise = (1. + np.sqrt(csi_noise)) * np.random.randn(H.shape)
            H = np.dot(H, curr_H_noise)
        return H

    @staticmethod
    def add_fading(H, fading, phase, n_user, iteration):
        if phase == 'test' and fading:
            degs_array = np.array([51, 39, 33, 21])
            fade_mat = (0.8 + 0.2 * np.cos(2 * np.pi * iteration / degs_array))
            fade_mat = np.tile(fade_mat.reshape(1, -1), [n_user, 1]).T
            H = H * fade_mat
        return H


class SEDChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user, iteration) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        return H


class GaussianChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user, iteration) -> np.ndarray:
        return np.random.randn(n_ant, n_user)


class COSTChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user, iteration) -> np.ndarray:
        total_h = np.empty([0, n_ant])
        for i in range(1, n_user + 1):
            path_to_mat = os.path.join(RESOURCES_DIR, f'h_omni_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['h_channel_response_mag'].T
            norm_h_user = (h_user - h_user.min()) / (h_user.max() - h_user.min())
            sampled_h_user = norm_h_user[np.linspace(0, norm_h_user.shape[0] - 1, n_ant, dtype=int), iteration]
            reshaped_sampled_h = sampled_h_user.reshape(-1, n_ant)
            # reshaped_sampled_h[0, :] /= 2
            # reshaped_sampled_h[0, i - 1] = 1
            total_h = np.concatenate([total_h, reshaped_sampled_h], axis=0)
        return total_h


if __name__ == "__main__":
    channel_model = ChannelModel()
    conf = Config()
    total_h = np.empty([conf.n_ant, conf.n_user, 0])
    for iteration in range(conf.test_frame_num):
        h = channel_model.get_channel(conf.channel_mode, conf.n_ant, conf.n_user,
                                      conf.csi_noise, 'test', conf.fading, iteration)
        total_h = np.concatenate([total_h, np.expand_dims(h, axis=2)], axis=2)

    i = 1
    for j in range(conf.n_user):
        plt.plot(total_h[i, j], label=f'{j}')
    plt.legend()
    plt.show()
