from python_code.utils.constants import Phase
from dir_definitions import RESOURCES_DIR
from typing import Union
import numpy as np
import scipy.io
import os

SCALING_COEF = 0.25
COST_CONFIG_FRAMES = 10
MAX_FRAMES = 40


class COSTChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, phase: Phase,
                          change_user_only: Union[int, None]) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        main_folder = (1 + (frame_ind % MAX_FRAMES) // COST_CONFIG_FRAMES)
        for i in range(1, n_user + 1):
            path_to_mat = os.path.join(RESOURCES_DIR, f'{phase.value}_{main_folder}', f'h_{i}.mat')
            h_user = scipy.io.loadmat(path_to_mat)['norm_channel'][frame_ind % COST_CONFIG_FRAMES]
            total_h[i - 1] = SCALING_COEF * h_user

            if change_user_only:
                H_row = np.array([i for i in range(n_ant)])
                H_row = np.tile(H_row, [n_user, 1]).T
                H_column = np.array([i for i in range(n_user)])
                H_column = np.tile(H_column, [n_ant, 1])
                H_sed = np.exp(-np.abs(H_row - H_column))
                remaining_indices = list(set(list(range(n_ant))) - set([change_user_only]))
                total_h[i - 1][remaining_indices] = H_sed[i - 1][remaining_indices]

        total_h[np.arange(n_user), np.arange(n_user)] = 1
        return total_h
