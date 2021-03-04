from python_code.data.data_generator import DataGenerator
from python_code.detectors.deep_sic_detector import DeepSICNet
from python_code.utils.config_singleton import Config
from python_code.utils.utils import symbol_to_prob, prob_to_symbol
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

HALF = 0.5


class Trainer:
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.train_dg = DataGenerator(conf.train_size)
        self.test_dg = DataGenerator(conf.test_size)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference

    def prepare_data_for_training(self, xs_train, ys_train, probs_vec):
        """
        Generates the DeepSIC Networks for Each User for the Iterations>1

        Parameters
        ----------
        data : dict
            The Data Dictionary Generated from DataGenerator class.

        Returns
        -------
        nets_list
            A list of length s_nK (number of users) containing instances of DeepSICNet for each user.
        v_cNet_m_fYtrain
            A list of data dictionaries with the prepard training data for each user
            [list_idx][dictionary_key]:
            [i]['x_train'] --> Training Labels (Symbol probabilities) for the i-th user.
            [i]['y_train'] --> Output of the Channel and the Predicted Symbol Probs. of the j-th users, where for j != i
        """
        nets_list = []
        x_train_all = []
        y_train_all = []
        for k in range(conf.K):
            idx = [i for i in range(conf.K) if i != k]
            x_train = symbol_to_prob(xs_train[:, k])
            y_train = torch.cat((ys_train, probs_vec[:, idx]), dim=1)
            x_train_all.append(x_train)
            y_train_all.append(y_train)
            nets_list.append(DeepSICNet())
        return nets_list, x_train_all, y_train_all

    def calculate_posteriors(self, trained_nets_list, i, probs_vec, y_train):
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for users in range(conf.K):
            idx = [i for i in range(conf.K) if i != users]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(trained_nets_list[users][i - 1](input))
            next_probs_vec[:, users] = output[:, 1].unsqueeze(-1)
        return next_probs_vec

    def train_model(self, net, x_train, y_train):
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
            out = net(y_train)
            loss = crt(out, x_train.squeeze(-1).long())
            loss.backward()
            opt.step()
        return net

    def train_models(self, trained_nets_list, i, networks_list, x_train_all, y_train_all):
        for user in range(conf.K):
            trained_nets_list[user][i] = self.train_model(networks_list[user],
                                                          x_train_all[user],
                                                          y_train_all[user])

    def evaluate(self, conf, trained_nets_list, snr):
        """
        Evaluates the performance of the model.

        Parameters
        ----------
        conf: an instance of the Conf class.
        trained_nets_list: 2D List
                A 2D list containing the optimized DeepSICNet Networks for each user per iteration,
                trained_nets_list[user_id][iteration]
        Returns
        -------
        BERs
            The Bit Error Rates/Ratios for the specified SNR of the testing dataset
        predicted_x
            The recovered symbols
        """
        x_test, y_test = self.test_dg(snr=snr)  # Generating data for the given SNR
        test_size, users_n, _ = x_test.shape
        predicted_x = torch.zeros(x_test.shape).to(device)
        for symbol in range(test_size):
            probs_vec = HALF * torch.ones(users_n, 1).unsqueeze(dim=0).to(device)
            cur_y_test = y_test[symbol].unsqueeze(dim=0)
            for i in range(conf.iterations):
                probs_vec = self.calculate_posteriors(trained_nets_list, i + 1, probs_vec, cur_y_test)
            predicted_x[symbol, :] = prob_to_symbol(probs_vec.float())
        print(f'Finished testing symbols')
        ber = (predicted_x != x_test).sum() / (torch.FloatTensor([test_size * users_n]))
        return predicted_x, ber.numpy()[0]

    def train(self):

        ber_list = []  # Contains the ber for each snr
        print(f'training')

        for snr in conf.snr_list:  # Traversing the SNRs
            print(f'snr {snr}')
            x_train, y_train = self.train_dg(snr=snr)  # Generating data for the given snr
            trained_nets_list = [[0] * conf.iterations for _ in
                                 range(conf.K)]  # 2D list for Storing the DeepSIC Networks
            initial_probs = symbol_to_prob(x_train)
            nets_list, x_train_all, y_train_all = self.prepare_data_for_training(x_train, y_train, initial_probs)

            # Training the DeepSIC network for each user for iteration=1
            self.train_models(trained_nets_list, 0, nets_list, x_train_all, y_train_all)

            # Initializing the probabilities
            probs_vec = HALF * torch.ones(x_train.shape).to(device)

            # Training the DeepSICNet for each user-symbol/iteration
            for i in range(1, conf.iterations):
                # Generating soft symbols for training purposes
                probs_vec = self.calculate_posteriors(trained_nets_list, i, probs_vec, y_train)

                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                nets_list, x_train_all, y_train_all = self.prepare_data_for_training(x_train, y_train, probs_vec)

                # Training the DeepSIC networks for the iteration>1
                self.train_models(trained_nets_list, i, nets_list, x_train_all, y_train_all)
            print('evaluating')
            # Testing the network on the current snr
            _, ber = self.evaluate(conf, trained_nets_list, snr)
            ber_list.append(ber)
            print(f'\nber :{ber} @ snr: {snr} [dB]')
        print(f'Training and Testing Completed\nBERs: {ber_list}')


if __name__ == "__main__":
    deep_sic_trainer = Trainer()
    deep_sic_trainer.train()
