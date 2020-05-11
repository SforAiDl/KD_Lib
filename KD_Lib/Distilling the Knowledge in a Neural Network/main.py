import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from train import train_large, train_distil
from evaluate import eval
from model import large, distil


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 100

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True)

    large_model = large().to(device)
    optimizer = optim.SGD(large_model.parameters(), lr=0.01, momentum=0.9)
    epochs = 20
    train_large(large_model, train_loader, optimizer, epochs, device)
    eval(large_model, test_loader)

    distil_model = distil().to(device)
    loss_fn = nn.MSELoss()
    distil_weight = 0.7
    temp = 20
    train_distil(large_model, distil_model, train_loader, optimizer, loss_fn, epochs, temp, distil_weight)
    eval(distil_model, test_loader)


if __name__ == '__main__':
    main()
