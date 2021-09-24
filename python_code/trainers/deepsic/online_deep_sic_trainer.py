from python_code.detectors.deep_sic_detector import DeepSICDetector
from python_code.trainers.deep_sic_trainer import DeepSICTrainer
from python_code.utils.config_singleton import Config
from torch import nn

import torch

from python_code.utils.constants import Phase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


class OnlineDeepSICTrainer(DeepSICTrainer):
    """
    Trainer for the DeepSIC model.
    """

    def __init__(self):
        super().__init__()
        self.self_supervised = True

    def __str__(self):
        return 'DeepSIC'

    def initialize_single_detector(self) -> nn.Module:
        """
        Loads the DeepSIC detector
        """
        return DeepSICDetector()

    def train_model(self, single_model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int):
        """
        Trains a DeepSIC Network

        Parameters
        ----------
        single_model: an instance of the DeepSICNet class to be trained.
        k_m_fYtrain:  dictionary
                      The training data dictionary to be used for optimizing the underlying DeepSICNet network.
        Returns
        -------
        k_DeepSICNet
            The optimized DeepSECNet network.
        """
        opt = torch.optim.Adam(single_model.parameters(), lr=conf.lr)
        crt = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(device)
        for _ in range(max_epochs):
            opt.zero_grad()
            out = single_model(y_train)
            loss = crt(out, b_train.squeeze(-1).long())
            loss.backward()
            opt.step()

    def online_train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int,
                          phase: Phase):
        self.train_loop(model, b_train, y_train, max_epochs, phase)


if __name__ == "__main__":
    trainer = OnlineDeepSICTrainer()
    trainer.main()
