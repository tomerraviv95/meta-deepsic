import torch


def symbol_to_prob(s):
    """
    symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
    Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
    :param s: symbols vector
    :return: probabilities vector
    """
    return 0.5 * (s + 1)


def prob_to_symbol(p):
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,0] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - 0.5)


def bpsk_modulate(b):
    """
    Generates a Tensor of Uniformly Distributed BPSK Symbols
    :return: a Tensor of Size: [Batch_Size, <# of RX_Antennas>, <# of Users>]
    """
    # generate symbols
    return (-1) ** b


def calculate_sigma_from_snr(SNR):
    """
    converts the Desired SNR into the noise power (noise variance)
    :param SNR: signal-to-noise ratio
    :return: noise's sigma
    """
    return 10 ** (-0.1 * SNR)
