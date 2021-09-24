from python_code.ecc.rs_main import ECC_BITS_PER_SYMBOL
from python_code.ecc.wrappers import decoder, encoder
from python_code.utils.constants import Phase
from python_code.utils.metrics import calculate_error_rates
from python_code.data.data_generator import DataGenerator
from python_code.utils.config_singleton import Config
from typing import List
import numpy as np
import random
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

SUBFRAMES_IN_FRAME = 5

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


class BlackBoxTrainer:
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.train_frame_size = conf.test_info_size if conf.use_ecc else conf.test_pilot_size
        self.test_frame_size = (
                conf.test_info_size + ECC_BITS_PER_SYMBOL * conf.n_ecc_symbols) if conf.use_ecc else conf.test_pilot_size
        self.train_dg = DataGenerator(conf.test_info_size, phase=Phase.TRAIN, frame_num=conf.train_frame_num)
        self.test_dg = DataGenerator(conf.test_info_size, phase=Phase.TEST, frame_num=conf.test_frame_num)
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.online_meta = False
        self.self_supervised = False
        self.phase = None

    def __str__(self):
        return 'trainer'

    def initialize_detector(self):
        pass

    def prepare_for_eval(self, c_frame_size, y_test):
        pass

    def train_model(self, net, x_train, y_train, max_epochs):
        pass

    def online_train_loop(self, x_train, y_train, model, max_epochs, phase):
        pass

    def train_loop(self, x_train, y_train, model, max_epochs, phase):
        pass

    def predict(self, model, y_test) -> torch.Tensor:
        pass

    def copy_detector(self, detector):
        return copy.deepcopy(detector)

    def get_word(self, current_x, ber, detected_word, encoded_word):
        if not conf.use_ecc:
            return current_x
        elif ber > 0:
            return detected_word
        return encoded_word

    def online_evaluate(self, model, snr) -> List[float]:
        # generate data and declare sizes
        b_test, y_test = self.test_dg(snr=snr)
        c_pred = torch.zeros_like(y_test)
        b_pred = torch.zeros_like(b_test)
        c_frame_size = c_pred.shape[0] // conf.test_frame_num
        b_frame_size = b_pred.shape[0] // conf.test_frame_num

        # saved detector is used to initialize the decoder in meta learning loops
        self.saved_detector = self.copy_detector(model)

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
                buffer_b, buffer_y = self.ecc_eval(buffer_b, buffer_y, ber_list, current_y, current_x, model, frame)
            else:
                buffer_b, buffer_y = self.pilot_eval(buffer_b, buffer_y, ber_list, current_y, current_x, model, frame)

        return ber_list

    def pilot_eval(self, buffer_b, buffer_y, ber_list, current_y, current_x, model, frame):

        x_pilot, x_data = current_x[:conf.test_pilot_size], current_x[conf.test_pilot_size:]
        y_pilot, y_data = current_y[:conf.test_pilot_size], current_y[conf.test_pilot_size:]

        last_x_pilot, last_y_pilot = buffer_b[-conf.test_pilot_size:], buffer_y[-conf.test_pilot_size:]

        # meta-learning main function
        if self.online_meta and (frame + 1) % SUBFRAMES_IN_FRAME == 0:
            print('Meta')
            self.train_loop(buffer_b, buffer_y, self.saved_detector, conf.self_supervised_epochs, self.phase)

        # use last word inserted in the buffer for training
        if self.self_supervised:
            if self.online_meta:
                model = self.copy_detector(self.saved_detector)
                self.online_train_loop(last_x_pilot, last_y_pilot, model, conf.self_supervised_epochs, self.phase)
            else:
                # use last word inserted in the buffer for training
                self.online_train_loop(x_pilot, y_pilot, model, conf.self_supervised_epochs, self.phase)

        # detect
        detected_word = self.predict(model, y_data)

        # save the encoded word in the buffer
        buffer_b = torch.cat([buffer_b, x_pilot], dim=0)
        buffer_y = torch.cat([buffer_y, y_pilot], dim=0)

        # calculate error rate
        ber = calculate_error_rates(detected_word, x_data)[0]
        ber_list.append(ber)
        print(frame, ber)

        return buffer_b, buffer_y

    def ecc_eval(self, buffer_b, buffer_y, ber_list, current_y, current_x, model, frame):
        # detect
        detected_word = self.predict(model, current_y)

        # decode
        decoded_word = decoder(detected_word, self.phase)

        # encode word again
        decoded_word_array = decoded_word.int().cpu().numpy()
        encoded_word = torch.Tensor(encoder(decoded_word_array, self.phase)).to(device)

        # calculate error rate
        ber = calculate_error_rates(decoded_word, current_x)[0]
        ber_list.append(ber)
        print(frame, ber)

        # save the encoded word in the buffer
        if ber <= conf.ber_thresh:
            est_word = self.get_word(current_x, ber, detected_word, encoded_word)
            buffer_b = torch.cat([buffer_b, est_word], dim=0)
            buffer_y = torch.cat([buffer_y, current_y], dim=0)

        # meta-learning main function
        if self.online_meta and (frame + 1) % SUBFRAMES_IN_FRAME == 0:
            print('Meta')
            self.train_loop(buffer_b, buffer_y, self.saved_detector, conf.self_supervised_epochs, self.phase)

        # use last word inserted in the buffer for training
        if self.self_supervised and ber <= conf.ber_thresh:
            if self.online_meta:
                model = self.copy_detector(self.saved_detector)
            # use last word inserted in the buffer for training
            self.online_train_loop(est_word, current_y, model, conf.self_supervised_epochs, self.phase)

        return buffer_b, buffer_y

    def agg_evaluate(self, model, snr):
        """
        Evaluates the performance of the model.

        Parameters
        ----------
        conf: an instance of the Conf class.
        model: 2D List
                A 2D list containing the optimized DeepSICNet Networks for each user per iteration,
                trained_nets_list[user_id][iteration]
        Returns
        -------
        BERs
            The Bit Error Rates/Ratios for the specified SNR of the testing dataset
        b_pred
            The recovered symbols
        """
        # generate data and declare sizes
        b_test, y_test = self.test_dg(snr=snr)  # Generating data for the given SNR
        c_pred = torch.zeros_like(y_test)
        b_pred = torch.zeros_like(b_test)
        c_frame_size = c_pred.shape[0] // conf.test_frame_num
        b_frame_size = b_pred.shape[0] // conf.test_frame_num

        c_pred = self.predict(model, y_test)
        print(f'Finished testing symbols')
        for frame in range(conf.test_frame_num - 1):
            c_start_ind = frame * c_frame_size
            c_end_ind = (frame + 1) * c_frame_size
            b_start_ind = frame * b_frame_size
            b_end_ind = (frame + 1) * b_frame_size
            b_pred[b_start_ind:b_end_ind] = decoder(c_pred[c_start_ind:c_end_ind], self.phase)
        ber = calculate_error_rates(b_pred, b_test)[0]
        return [ber]

    def evaluate(self, snr, model):
        print('evaluating')
        # Testing the network on the current snr
        if conf.eval_mode == 'by_word':
            ber = self.online_evaluate(model, snr)
        elif conf.eval_mode == 'aggregated':
            ber = self.agg_evaluate(model, snr)
        else:
            raise ValueError('No such evaluation mode!!!')
        return ber

    def train(self):
        all_bers = []  # Contains the ber
        print(f'training')
        print(f'snr {conf.snr}')
        self.phase = Phase.TRAIN
        b_train, y_train = self.train_dg(snr=conf.snr)  # Generating data for the given snr
        model = self.initialize_detector()
        self.train_loop(b_train, y_train, model, conf.max_epochs, self.phase)
        self.phase = Phase.TEST
        ber = self.evaluate(conf.snr, model)
        all_bers.append(ber)
        print(f'\nber :{sum(ber) / len(ber)} @ snr: {conf.snr} [dB]')
        print(f'Training and Testing Completed\nBERs: {all_bers}')
        return all_bers