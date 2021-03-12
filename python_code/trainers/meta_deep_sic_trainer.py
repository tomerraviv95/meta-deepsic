from python_code.detectors.meta_deep_sic_detector import MetaDeepSICDetector
from python_code.detectors.deep_sic_detector import DeepSICDetector
from python_code.utils.config_singleton import Config
from python_code.trainers.trainer import Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

MAML_FLAG = True
META_LR = 0.1


class MetaDeepSIC(object):
    pass


class MetaDeepSICTrainer(Trainer):
    """
    Trainer for the DeepSIC model.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Meta DeepSIC'

    def initialize_detector(self):
        """
        Loads the DeepSIC detector
        """
        return DeepSICDetector()

    def train_model(self, net, b_train, y_train):
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
        for _ in range(conf.max_epochs):
            opt.zero_grad()
            cur_support_idx = torch.arange(b_train.shape[0] - 1)
            cur_query_idx = torch.arange(1, b_train.shape[0])

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

        return net


if __name__ == "__main__":
    deep_sic_trainer = MetaDeepSICTrainer()
    deep_sic_trainer.train()
