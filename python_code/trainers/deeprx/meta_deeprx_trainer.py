from python_code.detectors.meta_deep_rx_detector import MetaDeepRXDetector
from python_code.detectors.deep_rx_detector import DeepRXDetector
from python_code.trainers.deeprx.rx_trainer import RXTrainer
from python_code.utils.config_singleton import Config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

MAML_FLAG = True
META_LR = 0.01
HALF = 0.5


class MetaDeepRXTrainer(RXTrainer):
    """
    Trainer for the meta DeepRX model.
    """

    def __init__(self):
        super().__init__()
        self.online_meta = True
        self.self_supervised = True

    def __str__(self):
        return 'Meta-DeepRX'

    def initialize_detector(self):
        """
        Loads the DeepRX detector
        """
        self.detector = DeepRXDetector()

    def train_model(self, net, x_train, y_train, max_epochs):
        """
        Main meta-training loop. Runs in minibatches, each minibatch is split to pairs of following words.
        The pairs are comprised of (support,query) words.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        opt = torch.optim.Adam(net.parameters(), lr=conf.lr)
        crt = torch.nn.BCELoss().to(device)
        m = torch.nn.Sigmoid()
        net = net.to(device)
        meta_detector = MetaDeepRXDetector()
        frame_size = self.train_frame_size if self.phase == 'train' else self.test_frame_size
        support_idx = torch.arange(x_train.shape[0] - frame_size)
        query_idx = torch.arange(frame_size, x_train.shape[0])
        META_SAMPLES = 25
        net.set_state('train')
        meta_detector.set_state('train')
        for _ in range(max_epochs):
            opt.zero_grad()

            # choose only META_SAMPLES samples from the entire support, query to use for current epoch
            cur_idx = torch.randperm(len(support_idx))[:META_SAMPLES]
            cur_support_idx, cur_query_idx = support_idx[cur_idx], query_idx[cur_idx]

            # divide the words to following pairs - (support,query)
            support_b, support_y = x_train[cur_support_idx], y_train[cur_support_idx]
            query_b, query_y = x_train[cur_query_idx], y_train[cur_query_idx]

            # local update (with support set)
            para_list_detector = list(map(lambda p: p[0], zip(net.parameters())))
            soft_estimation_supp = meta_detector(support_y, para_list_detector)
            loss_supp = crt(m(soft_estimation_supp), support_b)

            # set create_graph to True for MAML, False for FO-MAML
            local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=MAML_FLAG)
            updated_para_list_detector = list(
                map(lambda p: p[1] - META_LR * p[0], zip(local_grad, para_list_detector)))

            # meta-update (with query set) should be same channel with support set
            soft_estimation_query = meta_detector(query_y, updated_para_list_detector)
            loss_query = crt(m(soft_estimation_query), query_b)
            meta_grad = torch.autograd.grad(loss_query, para_list_detector, create_graph=False)

            ind_param = 0
            for param in net.parameters():
                param.grad = None  # zero_grad
                param.grad = meta_grad[ind_param]
                ind_param += 1

            opt.step()

    def online_train_loop(self, net, x_train, y_train, max_epochs):
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
        net.set_state('train')
        net = net.to(device)
        for _ in range(max_epochs):
            opt.zero_grad()
            out = net(y_train)
            loss = crt(input=m(out), target=x_train)
            loss.backward()
            opt.step()

    def predict(self, y_test):
        self.detector.set_state('test')
        return self.detector(y_test, self.train_frame_size if self.phase == 'train' else self.test_frame_size)

    def train_loop(self, x_train, y_train, max_epochs, phase):
        self.train_model(self.detector, x_train, y_train, max_epochs)


if __name__ == "__main__":
    deep_rx_trainer = MetaDeepRXTrainer()
    deep_rx_trainer.train()
