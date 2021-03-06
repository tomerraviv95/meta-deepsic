from python_code.utils.config_singleton import Config
from python_code.ecc.rs_main import decode, encode
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


def encoder(b, phase):
    if conf.use_ecc and phase == 'test':
        encoding = lambda b: encode(b, conf.n_ecc_symbols)
    else:
        encoding = lambda b: b

    return np.concatenate([encoding(b[:, i]).reshape(-1, 1) for i in range(b.shape[1])], axis=1)


def decoder(c_pred):
    if conf.use_ecc:
        decoding = lambda b: decode(b, conf.n_ecc_symbols)
    else:
        decoding = lambda b: b
    b_pred = np.zeros([conf.frame_num * conf.test_frame_size, conf.n_user])
    c_frame_size = c_pred.shape[0] // conf.frame_num
    b_frame_size = b_pred.shape[0] // conf.frame_num
    for i in range(conf.frame_num):
        for j in range(conf.n_user):
            b_pred[i * b_frame_size: (i + 1) * b_frame_size, j] = decoding(
                c_pred[i * c_frame_size: (i + 1) * c_frame_size, j].cpu().numpy())
    b_pred = torch.Tensor(b_pred).to(device)
    return b_pred
