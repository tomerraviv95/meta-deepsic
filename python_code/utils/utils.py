import torch


class Utils():
    """
    Utility class containing neccessary functions.
    Methods
    -------
    symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'

    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
        Converts Probabilities to BPSK Symbols by Hard Threhsolding: [0,0.5] -> '-1', [0.5,0] -> '+1'

    db_to_scalar(SNR:list)
        Converts the Desired SNR into the Noise Power (Noise Variance)
    """

    def symbol_to_prob(self, x):
        return 0.5 * (x + 1)

    def prob_to_symbol(self, x):
        return torch.sign(x - 0.5)

    def db_to_scalar(self, SNR):
        return torch.FloatTensor([10 ** (-0.1 * SNR)])
