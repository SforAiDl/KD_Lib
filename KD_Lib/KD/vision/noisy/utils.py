import torch


def add_noise(x, variance=0.1):
    """
    Function for adding gaussian noise

    :param x (torch.FloatTensor): Input for adding noise
    :param variance (float): Variance for adding noise
    """

    return x * (1 + (variance ** 0.5) * torch.randn_like(x))
