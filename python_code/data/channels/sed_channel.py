from python_code.utils.constants import Phase
from typing import Union
import numpy as np


class SEDChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, phase: Phase, frame_ind: int, fading: bool,
                          change_user_only: Union[int, None]) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        if fading and phase == Phase.TEST:
            H = SEDChannel.add_fading(H, phase, n_ant, frame_ind, change_user_only)

        return H

    @staticmethod
    def add_fading(H: np.ndarray, phase: Phase, n_ant: int, frame_ind: int,
                   change_user_only: Union[int, None]) -> np.ndarray:
        if phase == Phase.TRAIN:
            degs_array = np.array([1, 1, 1, 1])
            center = 1
        else:
            degs_array = np.array([51, 39, 33, 21])
            center = 0.8
        fade_mat = center + (1 - center) * np.cos(2 * np.pi * frame_ind / degs_array)
        if change_user_only:
            remaining_indices = list(set(list(range(n_ant))) - set([change_user_only]))
            fade_mat[remaining_indices] = 1
        fade_mat = np.tile(fade_mat.reshape(1, -1), [n_ant, 1])
        return H * fade_mat
