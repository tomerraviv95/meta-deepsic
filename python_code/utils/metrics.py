import torch
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_error_rates(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
    """
    Returns the ber,fer and error indices
    """
    prediction = prediction.long()
    target = target.long()
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    all_bits_sum_vector = torch.sum(torch.abs(prediction - target), 1).long()
    frames_acc = torch.eq(all_bits_sum_vector, torch.LongTensor(1).fill_(0).to(device=device)).float().mean().item()
    return max([1 - bits_acc, 0.0]), max([1 - frames_acc, 0.0]), torch.nonzero(all_bits_sum_vector,
                                                                               as_tuple=False).reshape(-1)
