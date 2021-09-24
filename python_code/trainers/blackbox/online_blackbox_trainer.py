from python_code.detectors.blackbox_detector import BlackBoxDetector
from python_code.trainers.blackbox_trainer import BlackBoxTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


class OnlineBlackBoxTrainer(BlackBoxTrainer):
    """
    Trainer for the online BlackBox model.
    """

    def __init__(self):
        super().__init__()
        self.self_supervised = True

    def __str__(self):
        return 'BlackBox'

    def initialize_single_detector(self) -> nn.Module:
        """
        Loads the BlackBox detector
        """
        return BlackBoxDetector()

    def train_model(self, single_model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int):
        """
        Trains the BlackBox Network

        Parameters
        ----------
        single_model: an instance of the DeepSICNet class to be trained.
        k_m_fYtrain:  dictionary
                      The training data dictionary to be used for optimizing the underlying DeepSICNet network.
        -------

        """
        opt = torch.optim.Adam(single_model.parameters(), lr=conf.lr)
        crt = torch.nn.BCELoss().to(device)
        m = torch.nn.Sigmoid()
        single_model.set_state(Phase.TRAIN)
        single_model = single_model.to(device)
        for _ in range(max_epochs):
            opt.zero_grad()
            out = single_model(y_train, self.train_frame_size if self.phase == Phase.TRAIN else self.test_frame_size)
            loss = crt(input=m(out), target=b_train.float())
            loss.backward()
            opt.step()

    def online_train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int,
                          phase: Phase):
        self.train_loop(model, b_train, y_train, max_epochs, phase)


if __name__ == "__main__":
    trainer = OnlineBlackBoxTrainer()
    trainer.main()
