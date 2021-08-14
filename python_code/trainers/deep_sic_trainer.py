from python_code.utils.utils import symbol_to_prob, prob_to_symbol
from python_code.utils.config_singleton import Config
from python_code.trainers.trainer import Trainer
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()
HALF = 0.5


class DeepSICTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'DeepSIC trainer'

    def initialize_single_detector(self):
        pass

    def prepare_data_for_training(self, x_train, ys_train, probs_vec):
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
        x_train_all = []
        y_train_all = []
        for k in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != k]
            y_train = torch.cat((ys_train, probs_vec[:, idx]), dim=1)
            x_train_all.append(x_train[:, k])
            y_train_all.append(y_train)
        return x_train_all, y_train_all

    def calculate_posteriors(self, trained_nets_list, i, probs_vec, y_train):
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for users in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != users]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(trained_nets_list[users][i - 1](input))
            next_probs_vec[:, users] = output[:, 1]
        return next_probs_vec

    def train_models(self, trained_nets_list, i, x_train_all, y_train_all, max_epochs, phase):
        for user in range(conf.n_user):
            if phase == 'test' and conf.retrain_user is not None:
                if not conf.retrain_user == user:
                    continue
            self.train_model(trained_nets_list[user][i], x_train_all[user], y_train_all[user], max_epochs)

    def train_loop(self, x_train, y_train, max_epochs, phase):
        initial_probs = x_train.clone()
        x_train_all, y_train_all = self.prepare_data_for_training(x_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, x_train_all, y_train_all, max_epochs, phase)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(x_train.shape).to(device)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, conf.iterations):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            x_train_all, y_train_all = self.prepare_data_for_training(x_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, x_train_all, y_train_all, max_epochs, phase)

    def initialize_detector(self):
        self.detector = [[self.initialize_single_detector() for _ in range(conf.iterations)]
                         for _ in range(conf.n_user)]  # 2D list for Storing the DeepSIC Networks

    def copy_detector(self, detector):
        return [copy.deepcopy(net) for net in detector]

    def predict(self, y_test):
        for i in range(conf.iterations):
            self.probs_vec = self.calculate_posteriors(self.detector, i + 1, self.probs_vec, y_test)
        c_pred = symbol_to_prob(prob_to_symbol(self.probs_vec.float()))
        return c_pred

    def prepare_for_eval(self, c_frame_size, y_test):
        if conf.use_ecc:
            self.probs_vec = HALF * torch.ones(c_frame_size, y_test.shape[1]).to(device)
        else:
            self.probs_vec = HALF * torch.ones(c_frame_size - conf.test_pilot_size, y_test.shape[1]).to(device)
