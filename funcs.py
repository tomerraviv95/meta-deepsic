from conf import Conf
from data_generator import DataGenerator
from deep_sic_detector import DeepSICNet
from utils import Utils
import matplotlib.pyplot as plt
import torch as tc


def GetICNetK(data):
    """
    Generates the DeepSIC Networks for Each User for the First Iteration

    Parameters
    ----------
    data : dict
        The Data Dictionary Generated from DataGenerator class.

    Returns
    -------
    v_cNet
        A list of length s_nK (number of users) containing instances of DeepSICNet for each user.
    v_cNet_m_fYtrain
        A list of data dictionaries with the prepard training data for each user
        [list_idx][dictionary_key]:
        [i]['m_fStrain'] --> Training Labels (Symbol probabilities) for the i-th user.
        [i]['m_fYtrain'] --> Output of the Channel and the Symbol Probs. of the j-th users, where for j != i

    """
    util = Utils()
    v_cNet = []
    v_cNet_m_fYtrain = []
    for k in range(conf.K):
        idx = [i for i in range(conf.K) if i != k]
        m_fStrain = util.fSymToProb(data['m_fStrain'][:, k])
        m_fYtrain = tc.cat((data['m_fYtrain'], util.fSymToProb(data['m_fStrain'][:, idx])), dim=1)
        k_data = {'m_fStrain': m_fStrain, 'm_fYtrain': m_fYtrain}
        v_cNet_m_fYtrain.append(k_data)
        v_cNet.append(DeepSICNet(conf, batch_size=conf.s_fTrainSize))
    return v_cNet, v_cNet_m_fYtrain


def GetICNet(data):
    """
    Generates the DeepSIC Networks for Each User for the Iterations>1

    Parameters
    ----------
    data : dict
        The Data Dictionary Generated from DataGenerator class.

    Returns
    -------
    v_cNet
        A list of length s_nK (number of users) containing instances of DeepSICNet for each user.
    v_cNet_m_fYtrain
        A list of data dictionaries with the prepard training data for each user
        [list_idx][dictionary_key]:
        [i]['m_fStrain'] --> Training Labels (Symbol robabilities) for the i-th user.
        [i]['m_fYtrain'] --> Output of the Channel and the Predicted Symbol Probs. of the j-th users, where for j != i
    """
    util = Utils()
    v_cNet = []
    v_cNet_m_fYtrain = []
    for k in range(conf.K):
        idx = [i for i in range(conf.K) if i != k]
        m_fStrain = util.fSymToProb(data['m_fStrain'][:, k])
        m_fYtrain = tc.cat((data['m_fYtrain'], data['m_fP'][:, idx]), dim=1)
        k_data = {'m_fStrain': m_fStrain, 'm_fYtrain': m_fYtrain}
        v_cNet_m_fYtrain.append(k_data)
        v_cNet.append(DeepSICNet(conf, batch_size=conf.s_fTrainSize))
    return v_cNet, v_cNet_m_fYtrain


def TrainICNet(k_DeepSICNet, k_m_fYtrain, user, iter_idx):
    """
    Trains a DeepSIC Network

    Parameters
    ----------
    k_DeepSICNet: an instance of the DeepSICNet class to be trained.
    k_m_fYtrain:  dictionary
                  The training data dictionary to be used for optimizing the underlying DeepSICNet network.
    Returns
    -------
    k_DeepSICNet
        The optimized DeepSECNet network.
    """
    opt = tc.optim.Adam(k_DeepSICNet.parameters(), lr=1e-2)
    crt = tc.nn.CrossEntropyLoss()
    max_epochs = 100
    for e in range(max_epochs):
        opt.zero_grad()
        out = k_DeepSICNet(k_m_fYtrain['m_fYtrain'])
        loss = crt(out, k_m_fYtrain['m_fStrain'].squeeze(-1).long())
        loss.backward()
        opt.step()
    return k_DeepSICNet


def s_fDetDeepSIC(conf, DeepSICs, data):
    """
    Evaluates the performance of the model.

    Parameters
    ----------
    conf: an instance of the Conf class.
    DeepSICs: 2D List
            A 2D list containing the optimized DeepSICNet Networks for each user per iteration,
            DeepSICs[user_id][iteration]
    Returns
    -------
    BERs
        The Bit Error Rates/Ratios for the specified SNR of the testing dataset
    m_fShat
        The recovered symbols
    """
    m_fY = data['m_fYtest']
    m_fS = data['m_fStest']
    s_nSymbols, s_nK, _ = m_fS.shape
    m_fShat = tc.zeros(m_fS.shape)
    softmax1 = tc.nn.Softmax(dim=0)
    for syms in range(s_nSymbols):
        v_fP = 0.5 * tc.ones(s_nK, 1)
        v_fPnext = tc.zeros(v_fP.shape)
        v_fCurY = m_fY[syms]
        for ii in range(conf.s_nIter):
            for kk in range(conf.K):
                idx = [i for i in range(conf.K) if i != kk]
                v_Input = tc.cat((v_fCurY, v_fP[idx]), dim=0)
                with tc.no_grad():
                    v_fPTemp = softmax1(DeepSICs[kk][ii](v_Input))
                v_fPnext[kk] = v_fPTemp[1].unsqueeze(-1)
            v_fP = v_fPnext
        m_fShat[syms, :] = util.fProbToSym(v_fP.float())
        if syms % int(s_nSymbols / 20) == 0:
            print(f'Testing | Symbols: {syms}/{s_nSymbols}', end='\r')
    print(f'Testing |Symbols: {syms}/{s_nSymbols}', end='\r')
    BER = (m_fShat != m_fS).sum() / (tc.FloatTensor([s_nSymbols * s_nK]))
    return m_fShat, BER


util = Utils()
conf = Conf()
DG = DataGenerator(conf)
softmax = tc.nn.Softmax(dim=1)  # Single symbol probability inference
softmax1 = tc.nn.Softmax(dim=0)  # Batch probability inference
BERs = []  # Contains the BER for each SNR
print(f'=====System Configuration=====\n\n{conf}')

for SNR in conf.v_fSNRdB:  # Traversing the SNRs
    data = DG(snr=SNR)  # Generating data for the given SNR
    DeepSICs = [[0] * conf.s_nIter for i in range(conf.K)]  # 2D list for Storing the DeepSIC Networks
    data_new = {}
    v_cNet, v_cNet_m_fYtrain = GetICNetK(data)  # Obtaining the DeepSIC networks and data for the first iteration

    # Training the DeepSIC network for each user for iteration=1
    for user in range(conf.K):
        DeepSICs[user][0] = TrainICNet(v_cNet[user], v_cNet_m_fYtrain[user], user, 0)

    # Initializing the probabilities
    m_fP = 0.5 * tc.ones(data['m_fStrain'].shape)

    # Training the DeepSICNet for each user-symbol/iteration
    for i in range(1, conf.s_nIter):
        m_fPNext = tc.zeros(data['m_fStrain'].shape)
        # Generating soft symbols for training purposes
        for users in range(conf.K):
            idx = [i for i in range(conf.K) if i != user]
            m_Input = tc.cat((data['m_fYtrain'], m_fP[:, idx]), dim=1)
            with tc.no_grad():
                m_fPTemp = softmax(DeepSICs[users][i - 1](m_Input))
            m_fPNext[:, users] = m_fPTemp[:, 1].unsqueeze(-1)
        m_fP = m_fPNext

        # Preparing the data to be fed into the DeepSIC networks for iteartions>1
        data_new['m_fStrain'] = data['m_fStrain']
        data_new['m_fYtrain'] = data['m_fYtrain']
        data_new['m_fP'] = m_fP

        # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
        v_cNet, v_cNet_m_fYtrain = GetICNet(data_new)

        # Training the DeepSIC networks for the iteration>1
        for user in range(conf.K):
            DeepSICs[user][i] = TrainICNet(v_cNet[user], v_cNet_m_fYtrain[user], user, i)

    # Testing the network on the current SNR
    _, BER = s_fDetDeepSIC(conf, DeepSICs, data)
    BERs.append(BER.numpy())
    print(f'\nBER :{BER} @ SNR: {SNR} [dB]')
print(f'Training and Testing Completed\nBERs: {BERs}')

fig1 = plt.figure(figsize=(6, 5))
ax = fig1.gca()

plt.semilogy(conf.v_fSNRdB, BERs, linestyle='--', marker='d', color='red', label=r'DeepSIC - Sequential - Perfect CSI',
             linewidth=2.5, fillstyle='none', markersize=10, markeredgewidth=2)

plt.title(r'DeepSIC: BPSK, $6\times 6$ MIMO System')
plt.ylabel(r'$\textbf{BER}$', fontsize=20)
plt.xlabel(r'SNR [dB]', fontsize=20)
plt.legend(handlelength=5)
plt.xticks(conf.v_fSNRdB)
plt.grid(True, which='both')
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

# fig1.savefig('BER_DeepSIC.pdf', dpi=60)
