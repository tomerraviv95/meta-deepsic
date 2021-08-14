from python_code.utils.config_singleton import Config
from torch.nn import functional as F
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
RESNET_PARAMS = 9

tanh_func = torch.nn.Tanh()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(MetaResnetBlock, self).__init__()

        self.skip = nn.Sequential()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=HIDDEN_SIZE, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Conv2d(in_channels=HIDDEN_SIZE, out_channels=out_channels, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor, var: list):
        identity = x

        if self.skip is not None:
            identity = F.batch_norm(F.conv2d(x, var[0], bias=None, padding=1, stride=3), var[2].detach(),
                                    var[1].detach(), weight=var[1], bias=var[2])

        # meta block
        relu = nn.ReLU()
        first = relu(F.batch_norm(F.conv2d(x, var[3], bias=None, padding=1, stride=1), var[5].detach(),
                                  var[4].detach(), weight=var[4], bias=var[5]))
        out = F.batch_norm(F.conv2d(first, var[6], bias=None, padding=1, stride=1), var[8].detach(),
                           var[7].detach(), weight=var[7], bias=var[8])

        out += identity
        out = tanh_func(out)

        return out


class MetaDeepRXDetector(nn.Module):
    """
    The DeepRXDetector Network Architecture
    """

    def __init__(self, total_frame_size):
        super(MetaDeepRXDetector, self).__init__()
        self.total_frame_size = total_frame_size
        self.all_blocks = [MetaResnetBlock(conf.n_ant, IN2_FILTERS),
                           MetaResnetBlock(IN2_FILTERS, IN3_FILTERS),
                           MetaResnetBlock(IN3_FILTERS, IN4_FILTERS),
                           MetaResnetBlock(IN4_FILTERS, IN5_FILTERS),
                           MetaResnetBlock(IN5_FILTERS, IN6_FILTERS),
                           MetaResnetBlock(IN6_FILTERS, IN7_FILTERS),
                           MetaResnetBlock(IN7_FILTERS, IN8_FILTERS),
                           MetaResnetBlock(IN8_FILTERS, IN9_FILTERS),
                           MetaResnetBlock(IN9_FILTERS, IN10_FILTERS),
                           MetaResnetBlock(IN10_FILTERS, conf.n_user)]
        self.state = None

    def reshaped_tensor_in(self, ten: torch.Tensor):
        return ten.reshape(-1, 1, conf.n_user, 1).transpose(dim0=1, dim1=2)

    def reshaped_tensor_out(self, ten: torch.Tensor):
        return ten.transpose(dim0=1, dim1=2).reshape(-1, conf.n_ant)

    def set_state(self, state: str):
        self.state = state

    def forward(self, y: torch.Tensor, var: list) -> torch.Tensor:
        cur_y = tanh_func(self.reshaped_tensor_in(y))
        for i in range(len(self.all_blocks)):
            cur_block = self.all_blocks[i]
            current_var = var[i * RESNET_PARAMS:(i + 1) * RESNET_PARAMS]
            cur_y = cur_block(cur_y, current_var)
        reshaped_out = self.reshaped_tensor_out(cur_y)
        if self.state == 'train':
            return reshaped_out
        # in eval mode
        elif self.state == 'test':
            m = torch.nn.Sigmoid()
            hard_decision_output = m(reshaped_out) >= 0.5
            return hard_decision_output
        else:
            raise Exception("No such state value!!!")
