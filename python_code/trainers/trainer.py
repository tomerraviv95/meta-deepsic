from python_code.data.data_generator import DataGenerator
from python_code.detectors.deep_sic_detector import DeepSICNet
from python_code.utils.config_singleton import Config
from python_code.utils.utils import symbol_to_prob, prob_to_symbol
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.conf = Config()
        self.DG = DataGenerator(self.conf)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.softmax1 = torch.nn.Softmax(dim=0)  # Batch probability inference

    def GetICNetK(self, data):
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
        v_cNet = []
        v_cNet_m_fYtrain = []
        for k in range(self.conf.K):
            idx = [i for i in range(self.conf.K) if i != k]
            m_fStrain = symbol_to_prob(data['m_fStrain'][:, k])
            m_fYtrain = torch.cat((data['m_fYtrain'], symbol_to_prob(data['m_fStrain'][:, idx])), dim=1)
            k_data = {'m_fStrain': m_fStrain, 'm_fYtrain': m_fYtrain}
            v_cNet_m_fYtrain.append(k_data)
            v_cNet.append(DeepSICNet(self.conf).to(device))
        return v_cNet, v_cNet_m_fYtrain

    def GetICNet(self, data):
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
        v_cNet = []
        v_cNet_m_fYtrain = []
        for k in range(self.conf.K):
            idx = [i for i in range(self.conf.K) if i != k]
            m_fStrain = symbol_to_prob(data['m_fStrain'][:, k])
            m_fYtrain = torch.cat((data['m_fYtrain'], data['m_fP'][:, idx]), dim=1)
            k_data = {'m_fStrain': m_fStrain, 'm_fYtrain': m_fYtrain}
            v_cNet_m_fYtrain.append(k_data)
            v_cNet.append(DeepSICNet(self.conf))
        return v_cNet, v_cNet_m_fYtrain

    def TrainICNet(self, k_DeepSICNet, k_m_fYtrain):
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
        opt = torch.optim.Adam(k_DeepSICNet.parameters(), lr=1e-2)
        crt = torch.nn.CrossEntropyLoss()
        max_epochs = 100
        for e in range(max_epochs):
            opt.zero_grad()
            out = k_DeepSICNet.to(device)(k_m_fYtrain['m_fYtrain'])
            loss = crt(out, k_m_fYtrain['m_fStrain'].squeeze(-1).long())
            loss.backward()
            opt.step()
        return k_DeepSICNet

    def evaluate(self, conf, DeepSICs, data):
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
        m_fShat = torch.zeros(m_fS.shape).to(device)
        softmax1 = torch.nn.Softmax(dim=0)
        for syms in range(s_nSymbols):
            v_fP = 0.5 * torch.ones(s_nK, 1).to(device)
            v_fPnext = torch.zeros(v_fP.shape).to(device)
            v_fCurY = m_fY[syms]
            for ii in range(conf.iterations):
                for kk in range(conf.K):
                    idx = [i for i in range(conf.K) if i != kk]
                    v_Input = torch.cat((v_fCurY, v_fP[idx]), dim=0)
                    with torch.no_grad():
                        v_fPTemp = softmax1(DeepSICs[kk][ii](v_Input))
                    v_fPnext[kk] = v_fPTemp[1].unsqueeze(-1)
                v_fP = v_fPnext
            m_fShat[syms, :] = prob_to_symbol(v_fP.float())
            if syms % int(s_nSymbols / 20) == 0:
                print(f'Testing | Symbols: {syms}/{s_nSymbols}', end='\r')
        print(f'Testing |Symbols: {syms}/{s_nSymbols}', end='\r')
        BER = (m_fShat != m_fS).sum() / (torch.FloatTensor([s_nSymbols * s_nK]))
        return m_fShat, BER

    def train(self):

        ber_list = []  # Contains the BER for each SNR
        print(f'training')

        for SNR in self.conf.snr_list:  # Traversing the SNRs
            print(f'SNR {SNR}')
            data = self.DG(snr=SNR)  # Generating data for the given SNR
            DeepSICs = [[0] * self.conf.iterations for i in
                        range(self.conf.K)]  # 2D list for Storing the DeepSIC Networks
            data_new = {}
            v_cNet, v_cNet_m_fYtrain = self.GetICNetK(
                data)  # Obtaining the DeepSIC networks and data for the first iteration

            # Training the DeepSIC network for each user for iteration=1
            for user in range(self.conf.K):
                DeepSICs[user][0] = self.TrainICNet(v_cNet[user], v_cNet_m_fYtrain[user], user, 0)

            # Initializing the probabilities
            m_fP = 0.5 * torch.ones(data['m_fStrain'].shape).to(device)

            # Training the DeepSICNet for each user-symbol/iteration
            for i in range(1, self.conf.iterations):
                print(i)
                m_fPNext = torch.zeros(data['m_fStrain'].shape).to(device)
                # Generating soft symbols for training purposes
                for users in range(self.conf.K):
                    idx = [i for i in range(self.conf.K) if i != user]
                    m_Input = torch.cat((data['m_fYtrain'], m_fP[:, idx]), dim=1)
                    with torch.no_grad():
                        m_fPTemp = self.softmax(DeepSICs[users][i - 1](m_Input))
                    m_fPNext[:, users] = m_fPTemp[:, 1].unsqueeze(-1)
                m_fP = m_fPNext

                # Preparing the data to be fed into the DeepSIC networks for iteartions>1
                data_new['m_fStrain'] = data['m_fStrain']
                data_new['m_fYtrain'] = data['m_fYtrain']
                data_new['m_fP'] = m_fP

                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                v_cNet, v_cNet_m_fYtrain = self.GetICNet(data_new)

                # Training the DeepSIC networks for the iteration>1
                for user in range(self.conf.K):
                    DeepSICs[user][i] = self.TrainICNet(v_cNet[user], v_cNet_m_fYtrain[user], user, i)
            print('evaluating')
            # Testing the network on the current SNR
            _, BER = self.evaluate(self.conf, DeepSICs, data)
            ber_list.append(BER.numpy())
            print(f'\nBER :{BER} @ SNR: {SNR} [dB]')
        print(f'Training and Testing Completed\nBERs: {ber_list}')


if __name__ == "__main__":
    deep_sic_trainer = Trainer()
    deep_sic_trainer.train()
