# -*- coding: utf-8 -*-
"""Tests for `KD_Lib` package."""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.TAKD.main import main_TAKD
from KD_Lib.original.mnist import mnist
from KD_Lib.noisy.main import noisy_mnist, noisy_cifar
from KD_Lib.models.resnet import (ResNet18,
                                  ResNet34,
                                  ResNet50,
                                  ResNet101,
                                  ResNet152)
from KD_Lib.attention.training import mnist as mnist_AT
from KD_Lib.original.original_paper import original
from KD_Lib.original.model import teacher, student
from KD_Lib.attention.attention import attention
from KD_Lib.noisy import NoisyTeacher


def test_noisy():
    noisy_mnist(epochs=0)
    noisy_cifar(num_classes=10, epochs=0)
    noisy_cifar(num_classes=100, epochs=0)

def test_original():
    mnist(epochs=0)

def test_resnet():
    params = [4, 4, 8, 8, 16]
    ResNet18(params)
    ResNet34(params)
    ResNet50(params)
    ResNet101(params)
    ResNet152(params)


def test_TAKD():
    config = {
        'teacher': {
            'name': 'resnet101',
            'params': [32, 32, 64, 64, 128],
            'optimizer': 'adam',
            'train_epoch': 0
        },
        'assistants': [
            {
                'name': 'resnet50',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam',
                'train_epoch': 0
            },
            {
                'name': 'resnet34',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam',
                'train_epoch': 0
            },
        ],
        'student': {
            'name': 'resnet18',
            'params': [16, 32, 32, 16, 8],
            'optimizer': 'adam',
            'train_epoch': 0
        },
        'dataset': {
            'name': 'mnist',
            'location': './data/mnist',
            'batch_size': 128,
            'num_classes': 10,
            'num_channels': 1
        },
        'loss_function': 'cross_entropy',
        'assistant_train_order': [[-1], [-1, 0]]
    }
    main_TAKD(config)


def test_RAKD():
    mnist(loss='RKD', epochs=0)


def test_attention_model():
    params = [4, 4, 8, 8, 16]
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    model = ResNet152(params, att=True)
    sample_output = model(sample_input)
    print(sample_output)


def test_AT():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    print(mnist_AT(teacher_params, student_params, epochs=0))

def test_original():
    teac = teacher(1200)
    stud = student(800)

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=False,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=32, shuffle=True)

    t_optimizer = optim.SGD(teac.parameters(), 0.01)
    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    orig = original(teac, stud, train_loader, test_loader, t_optimizer, s_optimizer, loss='RKD', rkd_angle=0.4, rkd_dist=0.6)

    orig.train_teacher(epochs=1,plot_losses=False,save_model=False)
    orig.train_student(epochs=1,plot_losses=False,save_model=False)
    orig.evaluate(teacher=False)

def test_attention():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10, True)
    student_model = ResNet18(student_params, 1, 10, True)

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=False,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=32, shuffle=True)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    att = attention(teacher_model, student_model, train_loader, test_loader, t_optimizer, s_optimizer,
                    loss='ATTENTION')

    att.train_teacher(epochs=0,plot_losses=False,save_model=False)
    att.train_student(epochs=0,plot_losses=False,save_model=False)
    att.evaluate(teacher=False)


def test_NoisyTeacher():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=32, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=False,
                           transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
            batch_size=32, shuffle=True)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    experiment = NoisyTeacher(teacher_model, student_model, train_loader,
                              test_loader, t_optimizer, s_optimizer,
                              alpha=0.4, noise_variance=0.2, device='cpu')

    experiment.train_teacher(epochs=0)
    experiment.train_student(epochs=0)
    experiment.evaluate(teacher=False)
