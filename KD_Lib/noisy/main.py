import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from .train import train_teacher, train_student
from .utils import eval, add_noise
from .model import ModLeNet, LeNet, NIN, Shallow
from KD_Lib.RKD import RKDLoss

def noisy_mnist(student_size=800, epochs=20, lr=0.01,
          optimizer='SGD', loss='MSE', batch_size=64,
          alpha=0.7, variance=1, rkd_angle=0.4, rkd_dist=0.6
         ):
    """
    Distill a linear student network (student_size) from a convolutional teacher
    network on the MNIST dataset.

    Keyword Arguments:
        student_size {int} -- Size of hidden layer in student model
                              (default: {800})
        epochs {int} -- Number of epochs to train each model (default: {20})
        lr {float} -- learning rate (default: {0.01})
        optimizer {str} -- optimizer to be used (supports SGD and adam)
                           (default: {'SGD'})
        loss {str} -- loss function to be used (supports mse, kl and rkd)
                      (rkd takes two additional arguments rkd_angle and
                      rkd_dist) (default: {'MSE'})
        batch_size {int} -- Batch Size to be used for training (default: {64})
        alpha {float} -- Probability with which training samples are selected 
                         for perturbation (default: {0.7})
        variance {float} -- Variance of the gaussian noise to be added
                            (default: {1})
        rkd_angle {float} -- Applicable only for rkd: ratio of angular
                             potential difference (default: {0.4})
        rkd_dist {float} -- Applicable only for rkd: ratio of distance
                             potential difference (default: {0.6})
    """

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

    teacher_model = LeNet(in_channels=1, img_size=28).to(device)
    student_model = Shallow(hidden_size=student_size).to(device)

    if optimizer.upper() == 'SGD':
        t_optimizer = optim.SGD(teacher_model.parameters(), lr, momentum=0.9)
        s_optimizer = optim.SGD(student_model.parameters(), lr, momentum=0.9)
    elif optimizer.upper() == 'Adam':
        t_optimizer = optim.Adam(teacher_model.parameters(), lr)
        s_optimizer = optim.Adam(student_model.parameters(), lr)

    if loss.upper() == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss.upper() == 'KL':
        loss_fn = nn.KLDivLoss()
    elif loss.upper() == 'RKD':
        loss_fn = RKDLoss(dist_ratio=rkd_dist, angle_ratio=rkd_angle)

    train_teacher(teacher_model, train_loader, t_optimizer, epochs)
    eval(teacher_model, test_loader)

    train_student(teacher_model, student_model, train_loader, s_optimizer,
                  loss_fn, epochs, alpha, variance)
    eval(student_model, test_loader)

def noisy_cifar(num_classes=10, epochs=20, lr=0.01,
          optimizer='SGD', loss='MSE', batch_size=64,
          alpha=0.7, variance=1, rkd_angle=0.4, rkd_dist=0.6
         ):
    """
    Distill a smaller convolutional student network from a larger teacher
    network on the CIFAR dataset.

    Keyword Arguments:
        num_classes {int} -- CIFAR10/CIFAR100 (default: {10})
        epochs {int} -- Number of epochs to train each model (default: {20})
        lr {float} -- learning rate (default: {0.01})
        optimizer {str} -- optimizer to be used (supports SGD and adam)
                           (default: {'SGD'})
        loss {str} -- loss function to be used (supports mse, kl and rkd)
                      (rkd takes two additional arguments rkd_angle and
                      rkd_dist) (default: {'MSE'})
        batch_size {int} -- Batch Size to be used for training (default: {64})
        alpha {float} -- Probability with which training samples are selected 
                         for perturbation (default: {0.7})
        variance {float} -- Variance of the gaussian noise to be added
                            (default: {1})
        rkd_angle {float} -- Applicable only for rkd: ratio of angular
                             potential difference (default: {0.4})
        rkd_dist {float} -- Applicable only for rkd: ratio of distance
                             potential difference (default: {0.6})
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if num_classes == 10:
        trainset = datasets.CIFAR10(root='cifar10_data', train=True, 
                                        download=True, 
                                        transform=transform_train)
        testset = datasets.CIFAR10(root='cifar10_data', train=False, 
                                        download=True, 
                                        transform=transform_test)

        teacher_model = NIN().to(device)
        student_model = ModLeNet().to(device)

    else:
        trainset = datasets.CIFAR100(root='cifar100_data', train=True, 
                                        download=True, 
                                        transform=transform_train)
        testset = datasets.CIFAR100(root='cifar100_data', train=False, 
                                        download=True, 
                                        transform=transform_test)
        teacher_model = NIN(num_classes=100).to(device)
        student_model = ModLeNet(num_classes=100).to(device)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    if optimizer.upper() == 'SGD':
        t_optimizer = optim.SGD(teacher_model.parameters(), lr, momentum=0.9)
        s_optimizer = optim.SGD(student_model.parameters(), lr, momentum=0.9)
    elif optimizer.upper() == 'Adam':
        t_optimizer = optim.Adam(teacher_model.parameters(), lr)
        s_optimizer = optim.Adam(student_model.parameters(), lr)

    if loss.upper() == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss.upper() == 'KL':
        loss_fn = nn.KLDivLoss()
    elif loss.upper() == 'RKD':
        loss_fn = RKDLoss(dist_ratio=rkd_dist, angle_ratio=rkd_angle)

    train_teacher(teacher_model, train_loader, t_optimizer, epochs)
    eval(teacher_model, test_loader)

    train_student(teacher_model, student_model, train_loader, s_optimizer,
                  loss_fn, epochs, alpha, variance)
    eval(student_model, test_loader)


if __name__ == '__main__':
    noisy_mnist()
    noisy_cifar()