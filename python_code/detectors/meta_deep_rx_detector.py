from python_code.utils.config_singleton import Config
from torch.nn import functional as F
from torch import nn
import torch

conf = Config()

RESNET_PARAMS = 9
RESNET_BLOCKS = 10

tanh_func = torch.nn.Tanh()
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

    def forward(self, x: torch.Tensor, var: list):
        identity = F.batch_norm(F.conv2d(x, var[0]), running_mean=var[2].detach(),
                                running_var=var[1].detach(), weight=var[1], bias=var[2])

        # meta block
        relu = nn.ReLU()
        first = relu(F.batch_norm(F.conv2d(x, var[3], bias=None, padding=1, stride=1), running_mean=var[5].detach(),
                                  running_var=var[4].detach(), weight=var[4], bias=var[5]))
        out = F.batch_norm(F.conv2d(first, var[6], bias=None, padding=1, stride=1), running_mean=var[8].detach(),
                           running_var=var[7].detach(), weight=var[7], bias=var[8])
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
        self.all_blocks = [MetaResnetBlock() for _ in range(RESNET_BLOCKS)]
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
