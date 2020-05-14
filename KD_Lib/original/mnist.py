import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from .train import train_teacher, train_student
from .evaluate import eval
from .model import teacher, student


def mnist(teacher_size=1200, student_size=800,epochs=20, lr=0.01, optimizer='SGD',
          loss='MSE', batch_size=100, distil_weight=0.7, temp=20 ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=batch_size, shuffle=True)  

    teacher_model = teacher(teacher_size).to(device)
    student_model = student(student_size).to(device)

    if optimizer.upper() == 'SGD':
        t_optimizer = optim.SGD(teacher_model.parameters(), lr, momentum=0.9)
        s_optimizer = optim.SGD(student_model.parameters(), lr,momentum=0.9)
    elif optimizer.upper() == 'Adam':
        t_optimizer = optim.Adam(teacher_model.parameters(), lr)
        s_optimizer = optim.Adam(student_model.parameters(), lr)
    
    if loss.upper() == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss.upper() == 'KL':
        loss_fn = nn.KLDivLoss()

    train_teacher(teacher_model, train_loader, t_optimizer, epochs)
    eval(teacher_model, test_loader)

    train_student(teacher_model, student_model, train_loader, s_optimizer, 
                  loss_fn, epochs, temp, distil_weight)
    eval(student_model, test_loader)


if __name__ == '__main__':
    mnist()
