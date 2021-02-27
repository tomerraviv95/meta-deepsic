from python_code.data.data_generator import DataGenerator
from python_code.detectors.deep_sic_detector import DeepSICNet
from python_code.utils.config_singleton import Config
from python_code.utils.utils import symbol_to_prob, prob_to_symbol
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


class Trainer:
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.train_dg = DataGenerator(conf.train_size)
        self.test_dg = DataGenerator(conf.test_size)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference

    def GetICNetK(self, x_train, y_train):
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
            [i]['x_train_probs'] --> Training Labels (Symbol probabilities) for the i-th user.
            [i]['y_train_probs'] --> Output of the Channel and the Symbol Probs. of the j-th users, where for j != i

        """
        v_cNet = []
        x_train_probs_all_users = []
        y_train_probs_all_users = []
        for k in range(conf.K):
            idx = [i for i in range(conf.K) if i != k]
            x_train_probs = symbol_to_prob(x_train[:, k])
            y_train_probs = torch.cat((y_train, symbol_to_prob(x_train[:, idx])), dim=1)
            x_train_probs_all_users.append(x_train_probs)
            y_train_probs_all_users.append(y_train_probs)
            v_cNet.append(DeepSICNet().to(device))
        return v_cNet, x_train_probs_all_users, y_train_probs_all_users

    def GetICNet(self, x_train, y_train, m_fP):
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
            [i]['x_train_probs'] --> Training Labels (Symbol probabilities) for the i-th user.
            [i]['y_train_probs'] --> Output of the Channel and the Predicted Symbol Probs. of the j-th users, where for j != i
        """
        v_cNet = []
        x_train_probs_all_users = []
        y_train_probs_all_users = []
        for k in range(conf.K):
            idx = [i for i in range(conf.K) if i != k]
            x_train_probs = symbol_to_prob(x_train[:, k])
            y_train_probs = torch.cat((y_train, m_fP[:, idx]), dim=1)
            x_train_probs_all_users.append(x_train_probs)
            y_train_probs_all_users.append(y_train_probs)
            v_cNet.append(DeepSICNet())
        return v_cNet, x_train_probs_all_users, y_train_probs_all_users

    def TrainICNet(self, net, x_train_probs, y_train_probs):
        """
        Trains a DeepSIC Network

        Parameters
        ----------
        net: an instance of the DeepSICNet class to be trained.
        k_m_fYtrain:  dictionary
                      The training data dictionary to be used for optimizing the underlying DeepSICNet network.
        Returns
        -------
        k_DeepSICNet
            The optimized DeepSECNet network.
        """
        opt = torch.optim.Adam(net.parameters(), lr=conf.lr)
        crt = torch.nn.CrossEntropyLoss()
        net = net.to(device)
        for _ in range(conf.max_epochs):
            opt.zero_grad()
            out = net(y_train_probs)
            loss = crt(out, x_train_probs.squeeze(-1).long())
            loss.backward()
            opt.step()
        return net

    def evaluate(self, conf, DeepSICs, SNR):
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
        m_fS, m_fY = self.test_dg(snr=SNR)  # Generating data for the given SNR
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

        for SNR in conf.snr_list:  # Traversing the SNRs
            print(f'SNR {SNR}')
            x_train, y_train = self.train_dg(snr=SNR)  # Generating data for the given SNR
            DeepSICs = [[0] * conf.iterations for _ in range(conf.K)]  # 2D list for Storing the DeepSIC Networks
            v_cNet, x_train_probs_all_users, y_train_probs_all_users = self.GetICNetK(x_train,
                                                                                      y_train)  # Obtaining the DeepSIC networks and data for the first iteration

            # Training the DeepSIC network for each user for iteration=1
            for user in range(conf.K):
                DeepSICs[user][0] = self.TrainICNet(v_cNet[user], x_train_probs_all_users[user],
                                                    y_train_probs_all_users[user])

            # Initializing the probabilities
            m_fP = 0.5 * torch.ones(x_train.shape).to(device)

            # Training the DeepSICNet for each user-symbol/iteration
            for i in range(1, conf.iterations):
                print(i)
                m_fPNext = torch.zeros(x_train.shape).to(device)
                # Generating soft symbols for training purposes
                for users in range(conf.K):
                    idx = [i for i in range(conf.K) if i != user]
                    m_Input = torch.cat((y_train, m_fP[:, idx]), dim=1)
                    with torch.no_grad():
                        m_fPTemp = self.softmax(DeepSICs[users][i - 1](m_Input))
                    m_fPNext[:, users] = m_fPTemp[:, 1].unsqueeze(-1)
                m_fP = m_fPNext

                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                v_cNet, x_train_probs_all_users, y_train_probs_all_users = self.GetICNet(x_train, y_train, m_fP)

                # Training the DeepSIC networks for the iteration>1
                for user in range(conf.K):
                    DeepSICs[user][i] = self.TrainICNet(v_cNet[user], x_train_probs_all_users[user],
                                                        y_train_probs_all_users[user])
            print('evaluating')
            # Testing the network on the current SNR
            _, BER = self.evaluate(conf, DeepSICs, SNR)
            ber_list.append(BER.numpy())
            print(f'\nBER :{BER} @ SNR: {SNR} [dB]')
        print(f'Training and Testing Completed\nBERs: {ber_list}')


if __name__ == "__main__":
    deep_sic_trainer = Trainer()
    deep_sic_trainer.train()
