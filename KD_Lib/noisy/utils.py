import torch


def add_noise(x, variance=0.1):
    return x * (1 + (variance**0.5) * torch.randn_like(x))
