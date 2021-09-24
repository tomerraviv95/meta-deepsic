from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()


class Trainer:
    def __init__(self):
        self.type = 'deepsic'

    def train(self):
        all_bers = []  # Contains the ber
        print(f'training')
        print(f'snr {conf.snr}')
        self.phase = Phase.TRAIN
        b_train, y_train = self.train_dg(snr=conf.snr)  # Generating data for the given snr
        if self.type == 'deepsic':
            trained_nets_list = [[self.initialize_detector() for _ in range(conf.iterations)]
                             for _ in range(conf.n_user)]  # 2D list for Storing the DeepSIC Networks
        else:
            trained_nets_list = self.initialize_detector_network()
        self.train_loop(b_train, y_train, trained_nets_list, conf.max_epochs, self.phase)
        self.phase = Phase.TEST
        ber = self.evaluate(conf.snr, trained_nets_list)
        all_bers.append(ber)
        print(f'\nber :{sum(ber) / len(ber)} @ snr: {conf.snr} [dB]')
        print(f'Training and Testing Completed\nBERs: {all_bers}')
        return all_bers
