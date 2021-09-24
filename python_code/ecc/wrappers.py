from python_code.utils.config_singleton import Config
from python_code.ecc.rs_main import decode, encode
from python_code.utils.constants import Phase
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


def encoder(b, phase):
    if phase == Phase.TEST and conf.use_ecc:
        encoding = lambda b: encode(b, conf.n_ecc_symbols)
    else:
        encoding = lambda b: b

    return np.concatenate([encoding(b[:, i]).reshape(-1, 1) for i in range(b.shape[1])], axis=1)


def decoder(c_pred, phase):
    if phase == Phase.TEST and conf.use_ecc:
        decoding = lambda b: decode(b, conf.n_ecc_symbols)
    else:
        decoding = lambda b: b
    b_pred = np.zeros([conf.test_info_size, conf.n_user])
    for j in range(conf.n_user):
        b_pred[:, j] = decoding(c_pred[:, j].cpu().numpy())
    b_pred = torch.Tensor(b_pred).to(device)
    return b_pred
