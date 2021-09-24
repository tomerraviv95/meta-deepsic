import pickle as pkl
import numpy as np
import torch


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str) -> np.ndarray:
    output = open(pkls_path, 'rb')
    return pkl.load(output)


def reshape_input(input_tensor: torch.Tensor, n_user: int, frame_size: int) -> torch.Tensor:
    return input_tensor.reshape(-1, frame_size, n_user, 1).transpose(dim0=1, dim1=2)


def reshape_output(output_tensor: torch.Tensor, n_ant: int) -> torch.Tensor:
    return output_tensor.transpose(dim0=1, dim1=2).reshape(-1, n_ant)
