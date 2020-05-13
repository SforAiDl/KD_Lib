import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from train import train_teacher, train_student
from evaluate import eval
from model import teacher, student

import argparse


def mnist():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='MNIST implementation of the "Distilling the Knowledge in a Neural Network"')

    parser.add_argument('--epochs', default=20, 
                        help='Default:20')
    parser.add_argument('--batch_size', default=100, 
                        help='Default:100')
    parser.add_argument('--lr', default=0.01,
                        help='Default:0.01')
    parser.add_argument('--loss', default='MSE', 
                        help='(Options:MSE,KL), (Default:MSE)')
    parser.add_argument('--optimizer', default='SGD',
                        help='(Options:SGD,Adam), (Default:SGD)')
    parser.add_argument('--teacher_size', default=1200, 
                        help='Default:1200')
    parser.add_argument('--student_size', default=800, 
                        help='Default:800')
    parser.add_argument('--distill_weight', default=0.7, 
                        help='Default:0.7')
    parser.add_argument('--temp', default=20, 
                        help='Default:20')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=args.batch_size, shuffle=True)  

    teacher_model = teacher(args.teacher_size).to(device)
    student_model = student(args.student_size).to(device)

    if args.optimizer == 'SGD':
        t_optimizer = optim.SGD(teacher_model.parameters(), args.lr, momentum=0.9)
        s_optimizer = optim.SGD(student_model.parameters(), args.lr,momentum=0.9)
    elif args.optimizer == 'Adam':
        t_optimizer = optim.Adam(teacher_model.parameters(),args.lr)
        s_optimizer = optim.Adam(student_model.parameters(),args.lr)
    
    if args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 'KL':
        loss_fn = nn.KLDivLoss()

    train_teacher(teacher_model, train_loader, t_optimizer, args.epochs, device)
    eval(teacher_model, test_loader)
    train_student(teacher_model, student_model, train_loader, s_optimizer, loss_fn, args.epochs, args.temp, args.distil_weight)
    eval(student_model, test_loader)


if __name__ == '__main__':
    mnist()
