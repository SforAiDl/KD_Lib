import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from KD_Lib.models.resnet import ResNet50, ResNet18
from .loss_metric import ATLoss
from .evaluate import eval


def train_teacher(model, train_loader, optimizer, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    loss_arr = []

    print('Training teacher - \n')

    for e in range(epochs):
        epoch_loss = 0
        for (data, label) in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)[0]
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1} loss = {epoch_loss}')


def train_student(teacher_model, student_model, train_loader, optimizer,
                  loss_fn, epochs=10, temp=20, distil_weight=0.7):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model.eval()
    student_model.train()
    loss_arr = []

    print('Training student - \n')

    for e in range(epochs):
        epoch_loss = 0

        for (data, label) in train_loader:

            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            teacher_out = teacher_model(data)
            student_out = student_model(data)

            soft_out = F.softmax(student_out[0]/temp, dim=1)

            loss = (1 - distil_weight) * F.cross_entropy(soft_out, label)
            loss += distil_weight * loss_fn(teacher_out, student_out)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1} loss = {epoch_loss}')


def mnist(teacher_parms, student_params, epochs=20, lr=0.01,
          optimizer='SGD', batch_size=100, distil_weight=0.7,
          temp=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True)

    teacher_model = ResNet50(teacher_parms, 1, 10, True).to(device)
    student_model = ResNet18(teacher_parms, 1, 10, True).to(device)

    if optimizer.upper() == 'SGD':
        t_optimizer = optim.SGD(teacher_model.parameters(), lr, momentum=0.9)
        s_optimizer = optim.SGD(student_model.parameters(), lr, momentum=0.9)
    elif optimizer.upper() == 'Adam':
        t_optimizer = optim.Adam(teacher_model.parameters(), lr)
        s_optimizer = optim.Adam(student_model.parameters(), lr)

    loss_fn = ATLoss()

    train_teacher(teacher_model, train_loader, t_optimizer, epochs)
    t_ac = eval(teacher_model, test_loader, device)

    train_student(teacher_model, student_model, train_loader, s_optimizer,
                  loss_fn, epochs, temp, distil_weight)
    s_ac = eval(student_model, test_loader, device)
    return t_ac, s_ac


if __name__ == '__main__':
    mnist()
