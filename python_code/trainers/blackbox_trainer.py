from python_code.trainers.trainer import Trainer
from python_code.utils.constants import Phase
from torch import nn
import torch
import copy


class BlackBoxTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'BlackBox Trainer'

    def initialize_model(self) -> nn.Module:
        return self.initialize_single_detector()

    def copy_model(self, model: nn.Module) -> nn.Module:
        return copy.deepcopy(model)

    def predict(self, model: nn.Module, y: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        model.set_state(Phase.TEST)
        return model(y, self.train_frame_size if self.phase == Phase.TRAIN else self.test_frame_size)

    def train_loop(self, model: nn.Module, b_train: torch.Tensor, y_train: torch.Tensor, max_epochs: int, phase: Phase):
        self.train_model(model, b_train, y_train, max_epochs)
