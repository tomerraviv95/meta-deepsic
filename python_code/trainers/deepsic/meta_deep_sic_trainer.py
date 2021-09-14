from python_code.detectors.meta_deep_sic_detector import MetaDeepSICDetector
from python_code.detectors.deep_sic_detector import DeepSICDetector
from python_code.utils.config_singleton import Config
from python_code.trainers.deepsic.deep_sic_trainer import DeepSICTrainer
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

MAML_FLAG = True
META_LR = 0.01
HALF = 0.5
META_SAMPLES = 1024


class MetaDeepSICDeepSICTrainer(DeepSICTrainer):
    """
    Trainer for the DeepSIC model.
    """

    def __init__(self):
        super().__init__()
        self.online_meta = True
        self.self_supervised = True

    def __str__(self):
        return 'Meta-DeepSIC'

    def initialize_detector(self):
        """
        Loads the DeepSIC detector
        """
        return DeepSICDetector()

    def train_model(self, net, b_train, y_train, max_epochs):
        """
        Main meta-training loop. Runs in minibatches, each minibatch is split to pairs of following words.
        The pairs are comprised of (support,query) words.
        Evaluates performance over validation SNRs.
        Saves weights every so and so iterations.
        """
        opt = torch.optim.Adam(net.parameters(), lr=conf.lr)
        crt = torch.nn.CrossEntropyLoss()
        net = net.to(device)
        meta_detector = MetaDeepSICDetector()
        support_idx = torch.arange(b_train.shape[0] - self.total_frame_size)
        query_idx = torch.arange(self.total_frame_size, b_train.shape[0])

        for _ in range(max_epochs):
            opt.zero_grad()

            # choose only META_SAMPLES samples from the entire support, query to use for current epoch
            cur_idx = torch.randperm(len(support_idx))[:META_SAMPLES]
            cur_support_idx, cur_query_idx = support_idx[cur_idx], query_idx[cur_idx]

            # divide the words to following pairs - (support,query)
            support_b, support_y = b_train[cur_support_idx], y_train[cur_support_idx]
            query_b, query_y = b_train[cur_query_idx], y_train[cur_query_idx]

            # local update (with support set)
            para_list_detector = list(map(lambda p: p[0], zip(net.parameters())))
            soft_estimation_supp = meta_detector(support_y, para_list_detector)
            loss_supp = crt(soft_estimation_supp, support_b.long())

            # set create_graph to True for MAML, False for FO-MAML
            local_grad = torch.autograd.grad(loss_supp, para_list_detector, create_graph=MAML_FLAG)
            updated_para_list_detector = list(
                map(lambda p: p[1] - META_LR * p[0], zip(local_grad, para_list_detector)))

            # meta-update (with query set) should be same channel with support set
            soft_estimation_query = meta_detector(query_y, updated_para_list_detector)
            loss_query = crt(soft_estimation_query, query_b.long())
            meta_grad = torch.autograd.grad(loss_query, para_list_detector, create_graph=False)

            ind_param = 0
            for param in net.parameters():
                param.grad = None  # zero_grad
                param.grad = meta_grad[ind_param]
                ind_param += 1

            opt.step()

    def online_train_loop(self, b_train, y_train, trained_nets_list, max_epochs, phase):
        trained_nets_list = [copy.deepcopy(net) for net in self.saved_nets_list]
        initial_probs = b_train.clone()
        b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.online_train_models(trained_nets_list, 0, b_train_all, y_train_all, max_epochs, phase)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(b_train.shape).to(device)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, conf.iterations):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(trained_nets_list, i, probs_vec, y_train)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            b_train_all, y_train_all = self.prepare_data_for_training(b_train, y_train, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.online_train_models(trained_nets_list, i, b_train_all, y_train_all, max_epochs, phase)

    def online_train_models(self, trained_nets_list, i, x_train_all, y_train_all, max_epochs, phase):
        for user in range(conf.n_user):
            if phase == 'test' and conf.retrain_user is not None:
                if not conf.retrain_user == user:
                    continue
            self.online_train_model(trained_nets_list[user][i], x_train_all[user], y_train_all[user], max_epochs)

    def online_train_model(self, net, x_train, y_train, max_epochs):
        opt = torch.optim.Adam(net.parameters(), lr=conf.lr)
        crt = torch.nn.CrossEntropyLoss()
        net = net.to(device)
        for _ in range(max_epochs):
            opt.zero_grad()
            out = net(y_train)
            loss = crt(out, x_train.squeeze(-1).long())
            loss.backward()
            opt.step()


if __name__ == "__main__":
    deep_sic_trainer = MetaDeepSICDeepSICTrainer()
    deep_sic_trainer.train()
