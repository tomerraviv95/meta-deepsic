import itertools
from python_code.trainers.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from typing import List
from torch import nn
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()
SEQUENTIAL_TRAINING = False


def symbol_to_prob(s: torch.Tensor) -> torch.Tensor:
    """
    symbol_to_prob(x:PyTorch/Numpy Tensor/Array)
    Converts BPSK Symbols to Probabilities: '-1' -> 0, '+1' -> '1.'
    :param s: symbols vector
    :return: probabilities vector
    """
    return HALF * (s + 1)


def prob_to_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,0] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


class DeepSICTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'DeepSIC Trainer'

    def initialize_model(self):
        return [[self.initialize_single_detector() for _ in range(conf.iterations)] for _ in
                range(conf.n_user)]  # 2D list for Storing the DeepSIC Networks

    def copy_model(self, model: nn.Module) -> List[nn.Module]:
        return [copy.deepcopy(single_model) for single_model in model]

    def train_models(self, model: nn.Module, i: int, b_train_all: torch.Tensor, y_train_all: torch.Tensor,
                     max_epochs: int, phase: Phase):
        for user in range(conf.n_user):
            if phase == Phase.TEST and conf.retrain_user is not None:
                if not conf.retrain_user == user:
                    continue
            self.train_model(model[user][i], b_train_all[user], y_train_all[user], max_epochs)

    def online_train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int,
                          phase: Phase):
        pass

    def train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int, phase: Phase):
        if SEQUENTIAL_TRAINING:
            self.sequential_train_loop(model, b_train, y_train, conf.max_epochs, self.phase)
        else:
            self.end_to_end_train_loop(model, b_train, y_train, conf.max_epochs, self.phase)

    def sequential_train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int,
                              phase: Phase):
        initial_probs = b_train.clone()
        b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(model, 0, b_train_all, y_train_all, max_epochs, phase)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(b_train.shape).to(device)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, conf.iterations):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(model, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(model, i, b_train_all, y_train_all, max_epochs, phase)

    def end_to_end_train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int,
                              phase: Phase):
        # Initializing the probabilities
        params = [j.parameters() for sub in model for j in sub]
        opt = torch.optim.Adam(itertools.chain(*params), lr=conf.lr)
        crt = torch.nn.CrossEntropyLoss()
        outputs = [[] for _ in range(conf.n_user)]
        for _ in range(max_epochs):
            for i in range(conf.iterations):
                # Generating soft symbols for training purposes
                if i > 0:
                    for user in range(conf.n_user):
                        idx = [i for i in range(conf.n_user) if i != user]
                        input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
                        with torch.no_grad():
                            output = self.softmax(model[user][i - 1](input))
                        probs_vec[:, user] = output[:, 1]
                    # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
                else:
                    probs_vec = b_train.clone()
                    # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)

                # Training the DeepSIC networks for the iteration>1
                for user in range(conf.n_user):
                    if phase == Phase.TEST and conf.retrain_user is not None:
                        if not conf.retrain_user == user:
                            continue
                    single_model, b, y = model[user][i].to(device), b_train_all[user], y_train_all[user]
                    output = single_model(y)
                    output_probs = self.softmax(output)
                    outputs[user].append(output)
                    probs_vec[:, user] = output_probs[:, 1]

                if i == 0:
                    probs_vec = HALF * torch.ones(b_train.shape).to(device)

            loss = 0
            for user in range(conf.n_user):
                loss += crt(outputs[user][-1], b_train_all[user].squeeze(-1).long())
            loss.backward()
            opt.step()
            opt.zero_grad()

    def predict(self, model: nn.Module, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        for i in range(conf.iterations):
            probs_vec = self.calculate_posteriors(model, i + 1, probs_vec, y)
        detected_word = symbol_to_prob(prob_to_symbol(probs_vec.float()))
        return detected_word

    def prepare_data_for_training(self, b_train: torch.Tensor, y_train: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
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
            [i]['b_train'] --> Training Labels (Symbol probabilities) for the i-th user.
            [i]['y_train'] --> Output of the Channel and the Predicted Symbol Probs. of the j-th users, where for j != i
        """
        b_train_all = []
        y_train_all = []
        for k in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != k]
            current_y_train = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            b_train_all.append(b_train[:, k])
            y_train_all.append(current_y_train)
        return b_train_all, y_train_all

    def calculate_posteriors(self, model: nn.Module, i: int, probs_vec: torch.Tensor,
                             y_train: torch.Tensor) -> torch.Tensor:
        next_probs_vec = torch.zeros(probs_vec.shape).to(device)
        for users in range(conf.n_user):
            idx = [i for i in range(conf.n_user) if i != users]
            input = torch.cat((y_train, probs_vec[:, idx]), dim=1)
            with torch.no_grad():
                output = self.softmax(model[users][i - 1](input))
            next_probs_vec[:, users] = output[:, 1]
        return next_probs_vec
