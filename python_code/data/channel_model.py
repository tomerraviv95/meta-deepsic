import torch


class ChannelModel:
    @staticmethod
    def calculate_channel(n_ant, n_user) -> torch.Tensor:
        pass

    @staticmethod
    def calculate_channel_wrapper(channel_mode, n_ant, n_user) -> torch.Tensor:
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
            curr_H_noise = (1. + torch.sqrt(torch.FloatTensor([csi_noise]))) * torch.randn(H.shape)
            H = torch.mul(H, curr_H_noise)
        return H


class SEDChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user) -> torch.Tensor:
        H_row = torch.FloatTensor([i for i in range(n_ant)])
        H_row = H_row.repeat([n_user, 1]).t()
        H_column = torch.FloatTensor([i for i in range(n_user)])
        H_column = H_column.repeat([n_ant, 1])
        H = torch.exp(-torch.abs(H_row - H_column))
        return H


class GaussianChannel(ChannelModel):
    @staticmethod
    def calculate_channel(n_ant, n_user) -> torch.Tensor:
        return torch.randn(n_ant, n_user)
