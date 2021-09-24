from python_code.detectors.blackbox_detector import BlackBoxDetector
from python_code.trainers.deeprx.rx_trainer import RXTrainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


class OnlineDeepRXTrainer(RXTrainer):
    """
    Trainer for the online DeepRX model.
    """

    def __init__(self):
        super().__init__()
        self.self_supervised = True

    def __str__(self):
        return 'DeepRX'

    def initialize_detector(self):
        """
        Loads the DeepRX detector
        """
        self.detector = BlackBoxDetector()

    def train_model(self, net, x_train, y_train, max_epochs):
        """
        Trains the DeepRX Network

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

    def predict(self, y_test):
        self.detector.set_state(Phase.TEST)
        return self.detector(y_test, self.train_frame_size if self.phase == Phase.TRAIN else self.test_frame_size)

    def train_loop(self, x_train, y_train, max_epochs, phase):
        self.train_model(self.detector, x_train, y_train, max_epochs)

    def online_train_loop(self, x_train, y_train, max_epochs, phase):
        self.train_loop(x_train, y_train, max_epochs, phase)


if __name__ == "__main__":
    deep_rx_trainer = OnlineDeepRXTrainer()
    deep_rx_trainer.train()
