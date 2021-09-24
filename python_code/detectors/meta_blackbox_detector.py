from python_code.utils.python_utils import reshape_input, reshape_output
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from torch.nn import functional as F
from torch import nn
import torch

conf = Config()

RESNET_PARAMS = 9
RESNET_BLOCKS = 10

tanh_func = torch.nn.Tanh()
relu_func = torch.nn.ReLU()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaResnetBlock(nn.Module):
    def __init__(self):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(MetaResnetBlock, self).__init__()

    def forward(self, x: torch.Tensor, var: list) -> torch.Tensor:
        skip_out = F.batch_norm(F.conv2d(x, var[0]), running_mean=var[2].detach(),
                                running_var=var[1].detach(), weight=var[1], bias=var[2])
        out1 = relu_func(F.batch_norm(F.conv2d(x, var[3], bias=None, padding=1, stride=1), running_mean=var[5].detach(),
                                      running_var=var[4].detach(), weight=var[4], bias=var[5]))
        out2 = F.batch_norm(F.conv2d(out1, var[6], bias=None, padding=1, stride=1), running_mean=var[8].detach(),
                            running_var=var[7].detach(), weight=var[7], bias=var[8])
        out = out2 + skip_out
        return tanh_func(out)


class MetaBlackBoxDetector(nn.Module):
    """
    The DeepRXDetector Network Architecture
    """

    def __init__(self):
        super(MetaBlackBoxDetector, self).__init__()
        self.all_blocks = [MetaResnetBlock() for _ in range(RESNET_BLOCKS)]
        self.state = None

    def set_state(self, state: Phase):
        self.state = state

    def forward(self, y: torch.Tensor, var: list) -> torch.Tensor:
        out = tanh_func(reshape_input(y, conf.n_user, 1))
        for i in range(len(self.all_blocks)):
            cur_block = self.all_blocks[i]
            current_var = var[i * RESNET_PARAMS:(i + 1) * RESNET_PARAMS]
            out = cur_block(out, current_var)
        out = reshape_output(out, conf.n_ant)
        if self.state == Phase.TRAIN:
            return out
        # in eval mode
        elif self.state == Phase.TEST:
            m = torch.nn.Sigmoid()
            hard_decision_output = m(out) >= HALF
            return hard_decision_output
        else:
            raise Exception("No such state value!!!")
