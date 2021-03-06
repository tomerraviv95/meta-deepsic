import numpy as np


class ChannelModel:
    @staticmethod
    def calculate_channel(n_ant, n_user) -> np.ndarray:
        pass

    @staticmethod
    def calculate_channel_wrapper(channel_mode, n_ant, n_user) -> np.ndarray:
        if channel_mode == 'SED':
            H = SEDChannel.calculate_channel(n_ant, n_user)
        elif channel_mode == 'Gaussian':
            H = GaussianChannel.calculate_channel(n_ant, n_user)
        else:
            raise NotImplementedError
        return H

    @staticmethod
    def get_channel(channel_mode, n_ant, n_user, csi_noise, phase):
        H = ChannelModel.calculate_channel_wrapper(channel_mode, n_ant, n_user)
        H = ChannelModel.noising_channel(H, csi_noise, phase)
        return H

    @staticmethod
    def noising_channel(H, csi_noise, phase):
        if phase == 'train' and csi_noise > 0:
            curr_H_noise = (1. + np.sqrt(csi_noise)) * np.random.randn(H.shape)
            H = np.dot(H, curr_H_noise)
        return H


class SEDChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row,[n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column,[n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        return H


class GaussianChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user) -> np.ndarray:
        return np.random.randn(n_ant, n_user)
