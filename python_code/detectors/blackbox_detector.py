from python_code.utils.python_utils import reshape_input, reshape_output
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, HALF
from torch import nn
import torch

conf = Config()

HIDDEN_SIZE = 100
IN2_FILTERS = 16
IN3_FILTERS = 32
IN4_FILTERS = 64
IN5_FILTERS = 128
IN6_FILTERS = 256
IN7_FILTERS = 128
IN8_FILTERS = 64
IN9_FILTERS = 32
IN10_FILTERS = 16

tanh_func = torch.nn.Tanh()


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResnetBlock, self).__init__()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=HIDDEN_SIZE, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Conv2d(in_channels=HIDDEN_SIZE, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_out = x
        out = self.block(x)

        if self.skip is not None:
            skip_out = self.skip(x)

        out += skip_out
        out = tanh_func(out)

        return out


class BlackBoxDetector(nn.Module):
    """
    The DeepRXDetector Network Architecture
    """

    def __init__(self):
        super(BlackBoxDetector, self).__init__()
        self.all_blocks = nn.Sequential(
            *[ResnetBlock(conf.n_ant, IN2_FILTERS),
              ResnetBlock(IN2_FILTERS, IN3_FILTERS),
              ResnetBlock(IN3_FILTERS, IN4_FILTERS),
              ResnetBlock(IN4_FILTERS, IN5_FILTERS),
              ResnetBlock(IN5_FILTERS, IN6_FILTERS),
              ResnetBlock(IN6_FILTERS, IN7_FILTERS),
              ResnetBlock(IN7_FILTERS, IN8_FILTERS),
              ResnetBlock(IN8_FILTERS, IN9_FILTERS),
              ResnetBlock(IN9_FILTERS, IN10_FILTERS),
              ResnetBlock(IN10_FILTERS, conf.n_user)]
        )
        self.state = None

    def set_state(self, state: Phase):
        self.state = state

    def forward(self, y: torch.Tensor, frame_size: int) -> torch.Tensor:
        reshaped_y_in = reshape_input(y, conf.n_user, frame_size)
        out = self.all_blocks(tanh_func(reshaped_y_in))
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
