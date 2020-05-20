import torch
import torch.optim as optim
import torch.nn as nn


def eval(model, data_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    total = len(data_loader.dataset)
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'{correct}/{total} correct')
    print(f'The accuracy of the model is {correct/total}')

def add_noise(x, variance = 0.1):
    return x * (1 + (variance**0.5) * torch.randn_like(x))

