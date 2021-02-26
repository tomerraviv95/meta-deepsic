import torch as tc

class Utils():
    """
    Utility class containing neccessary functions.
    Methods
    -------
    fSymToProb(x:PyTorch/Numpy Tensor/Array)
        Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'

    fProbToSym(x:PyTorch/Numpy Tensor/Array)
        Converts Probabilities to BPSK Symbols by Hard Threhsolding: [0,0.5] -> '-1', [0.5,0] -> '+1'

    fSNRToW(SNR:list)
        Converts the Desired SNR into the Noise Power (Noise Variance)
    """

    def fSymToProb(self, x):
        return 0.5 * (x + 1)

    def fProbToSym(self, x):
        return tc.sign(x - 0.5)

    def fSNRToW(self, SNR):
        return tc.FloatTensor([10 ** (-0.1 * SNR)])