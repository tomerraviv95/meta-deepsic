from python_code.detectors.blackbox_detector import BlackBoxDetector
from python_code.trainers.blackbox.blackbox_trainer import BlackBoxTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
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

    def initialize_detector(self):
        """
        Loads the BlackBox detector
        """
        return BlackBoxDetector()

    def train_model(self, net, x_train, y_train, max_epochs):
        """
        Trains the BlackBox Network

        Parameters
        ----------
        net: an instance of the DeepSICNet class to be trained.
        k_m_fYtrain:  dictionary
                      The training data dictionary to be used for optimizing the underlying DeepSICNet network.
        -------

        """
        opt = torch.optim.Adam(net.parameters(), lr=conf.lr)
        crt = torch.nn.BCELoss().to(device)
        m = torch.nn.Sigmoid()
        net.set_state(Phase.TRAIN)
        net = net.to(device)
        for _ in range(max_epochs):
            opt.zero_grad()
            out = net(y_train, self.train_frame_size if self.phase == Phase.TRAIN else self.test_frame_size)
            loss = crt(input=m(out), target=x_train.float())
            loss.backward()
            opt.step()

    def predict(self, model, y_test):
        model.set_state(Phase.TEST)
        return model(y_test, self.train_frame_size if self.phase == Phase.TRAIN else self.test_frame_size)

    def train_loop(self, x_train, y_train, model, max_epochs, phase):
        self.train_model(model, x_train, y_train, max_epochs)

    def online_train_loop(self, x_train, y_train, model, max_epochs, phase):
        self.train_loop(x_train, y_train, model, max_epochs, phase)


if __name__ == "__main__":
    deep_rx_trainer = OnlineBlackBoxTrainer()
    deep_rx_trainer.train()
