from python_code.ecc.rs_main import ECC_BITS_PER_SYMBOL
from python_code.ecc.wrappers import decoder, encoder
from python_code.utils.metrics import calculate_error_rates
from python_code.utils.utils import symbol_to_prob, prob_to_symbol
from python_code.data.data_generator import DataGenerator
from python_code.utils.config_singleton import Config
from typing import List
import numpy as np
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

torch.manual_seed(0)
np.random.seed(0)

HALF = 0.5
SUBFRAMES_IN_FRAME = 5


class DeepSICTrainer:
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.total_frame_size = conf.test_info_size + ECC_BITS_PER_SYMBOL * conf.n_ecc_symbols
        self.train_dg = DataGenerator(self.total_frame_size, phase='train', frame_num=conf.train_frame_num)
        self.test_dg = DataGenerator(conf.test_info_size, phase='test', frame_num=conf.test_frame_num)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.online_meta = False
        self.self_supervised = False

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
        b_train_all = []
        y_train_all = []
        for k in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != k]
            y_train = torch.cat((ys_train, probs_vec[:, idx]), dim=1)
            b_train_all.append(b_train[:, k])
            y_train_all.append(y_train)
        return b_train_all, y_train_all

    def calculate_posteriors(self, trained_nets_list, i, probs_vec, y_train):
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for users in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != users]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(trained_nets_list[users][i - 1](input))
            next_probs_vec[:, users] = output[:, 1]
        return next_probs_vec

    def train_model(self, net, x_train, y_train, max_epochs):
        pass

    def train_models(self, trained_nets_list, i, x_train_all, y_train_all, max_epochs, phase):
        for user in range(conf.n_user):
            if phase == 'test' and conf.retrain_user is not None:
                if not conf.retrain_user == user:
                    continue
            self.train_model(trained_nets_list[user][i], x_train_all[user], y_train_all[user], max_epochs)

    def online_train_loop(self, b_train, y_train, trained_nets_list, max_epochs, phase):
        pass

    def get_word(self, current_x, ber, detected_word, encoded_word):
        if not conf.use_ecc:
            return current_x
        elif ber > 0:
            return detected_word
        return encoded_word

    def online_evaluate(self, trained_nets_list, snr) -> List[float]:
        b_test, y_test = self.test_dg(snr=snr)  # Generating data for the given SNR
        c_pred = torch.zeros_like(y_test)
        b_pred = torch.zeros_like(b_test)
        c_frame_size = c_pred.shape[0] // conf.test_frame_num
        b_frame_size = b_pred.shape[0] // conf.test_frame_num
        if conf.use_ecc:
            probs_vec = HALF * torch.ones(c_frame_size, y_test.shape[1]).to(device)
        else:
            probs_vec = HALF * torch.ones(c_frame_size - conf.test_pilot_size, y_test.shape[1]).to(device)

        # saved detector is used to initialize the decoder in meta learning loops
        self.saved_nets_list = [copy.deepcopy(net) for net in trained_nets_list]

        # query for all detected words
        buffer_b, buffer_y = torch.empty([0, b_test.shape[1]]).to(device), torch.empty([0, y_test.shape[1]]).to(device)

        ber_list = []
        for frame in range(conf.test_frame_num - 1):
            # current word
            c_start_ind = frame * c_frame_size
            c_end_ind = (frame + 1) * c_frame_size
            current_y = y_test[c_start_ind:c_end_ind]
            b_start_ind = frame * b_frame_size
            b_end_ind = (frame + 1) * b_frame_size
            current_x = b_test[b_start_ind:b_end_ind]

            if conf.use_ecc:
                buffer_b, buffer_y = self.ecc_eval(buffer_b, buffer_y, probs_vec, ber_list, current_y, current_x,
                                                   trained_nets_list, frame)
            else:
                buffer_b, buffer_y = self.pilot_eval(buffer_b, buffer_y, probs_vec, ber_list, current_y, current_x,
                                                     trained_nets_list, frame)

        return ber_list

    def pilot_eval(self, buffer_b, buffer_y, probs_vec, ber_list, current_y, current_x, trained_nets_list, frame):

        x_pilot, x_data = current_x[:conf.test_pilot_size], current_x[conf.test_pilot_size:]
        y_pilot, y_data = current_y[:conf.test_pilot_size], current_y[conf.test_pilot_size:]

        # save the encoded word in the buffer
        buffer_b = torch.cat([buffer_b, x_pilot], dim=0)
        buffer_y = torch.cat([buffer_y, y_pilot], dim=0)

        # meta-learning main function
        if self.online_meta and (frame + 1) % SUBFRAMES_IN_FRAME == 0:
            print('Meta')
            # initialize from trained weights
            self.train_loop(buffer_b, buffer_y, self.saved_nets_list, conf.self_supervised_epochs, 'test')
            trained_nets_list = [copy.deepcopy(net) for net in self.saved_nets_list]

        # use last word inserted in the buffer for training
        if self.self_supervised:
            # use last word inserted in the buffer for training
            self.online_train_loop(x_pilot, y_pilot, trained_nets_list,
                                   conf.self_supervised_epochs, 'test')

        # detect and decode
        for i in range(conf.iterations):
            probs_vec = self.calculate_posteriors(trained_nets_list, i + 1, probs_vec, y_data)
        detected_word = symbol_to_prob(prob_to_symbol(probs_vec.float()))

        # calculate error rate
        ber = calculate_error_rates(detected_word, x_data)[0]
        ber_list.append(ber)
        print(frame, ber)

        return buffer_b, buffer_y

    def ecc_eval(self, buffer_b, buffer_y, probs_vec, ber_list, current_y, current_x, trained_nets_list, frame):

        # detect and decode
        for i in range(conf.iterations):
            probs_vec = self.calculate_posteriors(trained_nets_list, i + 1, probs_vec, current_y)
        detected_word = symbol_to_prob(prob_to_symbol(probs_vec.float()))

        # decode
        decoded_word = decoder(detected_word, 'test')

        # encode word again
        decoded_word_array = decoded_word.int().cpu().numpy()
        encoded_word = torch.Tensor(encoder(decoded_word_array, 'test')).to(device)

        # calculate error rate
        ber = calculate_error_rates(decoded_word, current_x)[0]
        ber_list.append(ber)
        print(frame, ber)

        # save the encoded word in the buffer
        if ber <= conf.ber_thresh:
            to_buffer_word = self.get_word(current_x, ber, detected_word, encoded_word)
            buffer_b = torch.cat([buffer_b, to_buffer_word], dim=0)
            buffer_y = torch.cat([buffer_y, current_y], dim=0)

        # meta-learning main function
        if self.online_meta and (frame + 1) % SUBFRAMES_IN_FRAME == 0:
            print('Meta')
            # initialize from trained weights
            self.train_loop(buffer_b, buffer_y, self.saved_nets_list, conf.self_supervised_epochs, 'test')
            trained_nets_list = [copy.deepcopy(net) for net in self.saved_nets_list]

        # use last word inserted in the buffer for training
        if self.self_supervised and ber <= conf.ber_thresh:
            # use last word inserted in the buffer for training
            self.online_train_loop(to_buffer_word, current_y, trained_nets_list, conf.self_supervised_epochs, 'test')

        return buffer_b, buffer_y

    def agg_evaluate(self, trained_nets_list, snr):
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
        b_pred = torch.zeros_like(b_test)
        c_frame_size = c_pred.shape[0] // conf.test_frame_num
        b_frame_size = b_pred.shape[0] // conf.test_frame_num
        for frame in range(conf.test_frame_num - 1):
            c_start_ind = frame * c_frame_size
            c_end_ind = (frame + 1) * c_frame_size
            b_start_ind = frame * b_frame_size
            b_end_ind = (frame + 1) * b_frame_size
            b_pred[b_start_ind:b_end_ind] = decoder(c_pred[c_start_ind:c_end_ind], 'test')
        ber = calculate_error_rates(b_pred, b_test)[0]
        return [ber]

    def evaluate(self, snr, trained_nets_list):
        print('evaluating')
        # Testing the network on the current snr
        if conf.eval_mode == 'by_word':
            ber = self.online_evaluate(trained_nets_list, snr)
        elif conf.eval_mode == 'aggregated':
            ber = self.agg_evaluate(trained_nets_list, snr)
        else:
            raise ValueError('No such evaluation mode!!!')
        return ber

    def train_loop(self, b_train, y_train, trained_nets_list, max_epochs, phase):
        initial_probs = b_train.clone()
        b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(trained_nets_list, 0, b_train_all, y_train_all, max_epochs, phase)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(b_train.shape).to(device)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, conf.iterations):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(trained_nets_list, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(trained_nets_list, i, b_train_all, y_train_all, max_epochs, phase)

    def train(self):
        all_bers = []  # Contains the ber
        print(f'training')
        print(f'snr {conf.snr}')
        b_train, y_train = self.train_dg(snr=conf.snr)  # Generating data for the given snr
        trained_nets_list = [[self.initialize_detector() for _ in range(conf.iterations)]
                             for _ in range(conf.n_user)]  # 2D list for Storing the DeepSIC Networks
        self.train_loop(b_train, y_train, trained_nets_list, conf.max_epochs, 'train')
        ber = self.evaluate(conf.snr, trained_nets_list)
        all_bers.append(ber)
        print(f'\nber :{sum(ber) / len(ber)} @ snr: {conf.snr} [dB]')
        print(f'Training and Testing Completed\nBERs: {all_bers}')
        return all_bers
