from python_code.detectors.deep_sic_detector import DeepSICDetector
from python_code.trainers.trainer import Trainer
from python_code.utils.config_singleton import Config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


class JointDeepSICTrainer(Trainer):
    """
    Trainer for the DeepSIC model.
    """

    def __init__(self):
        super().__init__()
        self.self_supervised = False

    def __str__(self):
        return 'DeepSIC'

    def initialize_detector(self):
        """
        Loads the DeepSIC detector
        """
        return DeepSICDetector()

    def train_model(self, net, x_train, y_train, max_epochs):
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
        for _ in range(max_epochs):
            opt.zero_grad()
            out = net(y_train)
            loss = crt(out, x_train.squeeze(-1).long())
            loss.backward()
            opt.step()

    def online_train_loop(self, b_train, y_train, trained_nets_list, max_epochs):
        self.train_loop(b_train, y_train, trained_nets_list, max_epochs)


if __name__ == "__main__":
    deep_sic_trainer = JointDeepSICTrainer()
    deep_sic_trainer.train()
