import torch as tc
tc.manual_seed(1)
import matplotlib.pyplot as plt
from dataclasses import *
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
plt.rc('font', weight='bold')
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

@dataclass(frozen=True)  # Immutable Data Class
class Conf:
    """Form the configuration class.

    Keyword arguments:
    N --- Number of Rx Antennas
    s_nK --- Number of Transmitted Symbols
    v_fSNRdB --- SNR Values in dB
    s_nIter --- Interference Cancellation Iterations
    s_fTrainSize --- Training Data Size
    s_fTestSize --- Test Data Size
    s_fEstErrVar --- Estimation Error Variance
    s_fFrameSize --- Frame Size for Generating Noisy Training Data
    s_fNumFrames --- Number of Frames
    v_nCurves --- A Binary List Indicating the Simulated Algorithms [Soft IC, perfect CSI,
                                                                     Soft IC, CSI uncertainty,
                                                                     Sequential DeepSIC, Perfect CSI,
                                                                     Sequential DeepSIC, CSI Uncertainty]

    ChannelModel --- The Channel Model: set 'SEQ' for the Spatial Exponential Decay Channel Model, i.e. exp(-|i-j|),
                                        set 'Gaussian' for a Gaussian Channel, i.e. N(0,1).
    """
    N: int = 6
    K: int = 6
    v_fSNRdB: list = field(default_factory=lambda: [i for i in range(0, 16, 2)])
    s_nIter: int = 5
    s_fTrainSize: int = 5000
    s_fTestSize: int = 20000
    s_fEstErrVar: float = 0.1
    s_fFrameSize: int = 500
    s_fNumFrames: int = int(s_fTrainSize / s_fFrameSize)
    v_nCurves: list = field(default_factory=lambda: [1, 1, 1, 1])  # This variable is depricated
    s_nCruves: int = 4
    ChannelModel: str = 'SED'
    v_fConst: list = field(default_factory=lambda: [-1, 1])
    DeepSICArch: str = 'MLP'