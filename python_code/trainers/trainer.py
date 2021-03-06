from python_code.ecc.wrappers import decoder
from python_code.utils.metrics import calculate_error_rates
from python_code.utils.utils import symbol_to_prob, prob_to_symbol
from python_code.data.data_generator import DataGenerator
from python_code.utils.config_singleton import Config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

HALF = 0.5


class Trainer:
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.train_dg = DataGenerator(conf.train_frame_size, phase='train')
        self.test_dg = DataGenerator(conf.test_frame_size, phase='test')
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.initialize_detector()

    def __str__(self):
        return 'trainer'

    def initialize_detector(self):
        pass

    def prepare_data_for_training(self, b_train, ys_train, probs_vec):
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
        b_train_all = []
        y_train_all = []
        for k in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != k]
            y_train = torch.cat((ys_train, probs_vec[:, idx]), dim=1)
            b_train_all.append(b_train[:, k])
            y_train_all.append(y_train)
            nets_list.append(self.initialize_detector())
        return nets_list, b_train_all, y_train_all

    def calculate_posteriors(self, trained_nets_list, i, probs_vec, y_train):
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for users in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != users]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(trained_nets_list[users][i - 1](input))
            next_probs_vec[:, users] = output[:, 1]
        return next_probs_vec

    def train_model(self, net, x_train, y_train):
        pass

    def train_models(self, trained_nets_list, i, networks_list, x_train_all, y_train_all):
        for user in range(conf.n_user):
            trained_nets_list[user][i] = self.train_model(networks_list[user],
                                                          x_train_all[user],
                                                          y_train_all[user])

    def online_evaluate(self, trained_nets_list, snr):
        b_test, y_test = self.test_dg(snr=snr)  # Generating data for the given SNR
        c_pred = torch.zeros_like(y_test)
        b_pred = torch.zeros_like(b_test)
        c_frame_size = c_pred.shape[0] // conf.frame_num
        b_frame_size = b_pred.shape[0] // conf.frame_num
        probs_vec = HALF * torch.ones(c_frame_size, y_test.shape[1]).to(device)
        total_ber = 0
        for frame in range(conf.frame_num):
            c_start_ind = frame * c_frame_size
            b_start_ind = frame * b_frame_size
            c_end_ind = (frame + 1) * c_frame_size
            b_end_ind = (frame + 1) * b_frame_size
            current_y = y_test[c_start_ind:c_end_ind]
            for i in range(conf.iterations):
                probs_vec = self.calculate_posteriors(trained_nets_list, i + 1, probs_vec, current_y)
            c_pred[c_start_ind:c_end_ind] = symbol_to_prob(prob_to_symbol(probs_vec.float()))
            b_pred[b_start_ind:b_end_ind] = decoder(c_pred[c_start_ind:c_end_ind])
            ber = calculate_error_rates(b_pred[b_start_ind:b_end_ind], b_test[b_start_ind:b_end_ind])[0]
            total_ber += ber

            if conf.self_supervised and ber <= conf.ber_thresh:
                # use last word inserted in the buffer for training
                self.online_training()

            print(frame, ber)
        return total_ber / conf.frame_num

    def evaluate(self, trained_nets_list, snr):
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
        b_pred
            The recovered symbols
        """
        b_test, y_test = self.test_dg(snr=snr)  # Generating data for the given SNR
        probs_vec = HALF * torch.ones(y_test.shape).to(device)
        for i in range(conf.iterations):
            probs_vec = self.calculate_posteriors(trained_nets_list, i + 1, probs_vec, y_test)
        c_pred = symbol_to_prob(prob_to_symbol(probs_vec.float()))
        print(f'Finished testing symbols')
        b_pred = decoder(c_pred)
        ber = calculate_error_rates(b_pred, b_test)[0]
        return ber

    def train(self):

        ber_list = []  # Contains the ber for each snr
        print(f'training')
        for snr in conf.snr_list:  # Traversing the SNRs
            print(f'snr {snr}')
            b_train, y_train = self.train_dg(snr=snr)  # Generating data for the given snr
            trained_nets_list = [[0] * conf.iterations for _ in
                                 range(conf.n_user)]  # 2D list for Storing the DeepSIC Networks
            initial_probs = b_train.clone()
            nets_list, b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)

            # Training the DeepSIC network for each user for iteration=1
            self.train_models(trained_nets_list, 0, nets_list, b_train_all, y_train_all)

            # Initializing the probabilities
            probs_vec = HALF * torch.ones(b_train.shape).to(device)

            # Training the DeepSICNet for each user-symbol/iteration
            for i in range(1, conf.iterations):
                # Generating soft symbols for training purposes
                probs_vec = self.calculate_posteriors(trained_nets_list, i, probs_vec, y_train)

                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                nets_list, b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)

                # Training the DeepSIC networks for the iteration>1
                self.train_models(trained_nets_list, i, nets_list, b_train_all, y_train_all)
            print('evaluating')
            # Testing the network on the current snr
            ber = self.online_evaluate(trained_nets_list, snr)
            ber_list.append(ber)
            print(f'\nber :{ber} @ snr: {snr} [dB]')
        print(f'Training and Testing Completed\nBERs: {ber_list}')
        return ber_list
