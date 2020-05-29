# -*- coding: utf-8 -*-
"""Tests for `KD_Lib` package."""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.models.resnet import (ResNet18,
                                  ResNet34,
                                  ResNet50,
                                  ResNet101,
                                  ResNet152)
from KD_Lib.TAKD.takd import TAKD
from KD_Lib.attention.attention import attention
from KD_Lib.original.original_paper import original
from KD_Lib.noisy import NoisyTeacher
from KD_Lib.Bert2Lstm.utils import get_essentials
from KD_Lib.Bert2Lstm.bert2lstm import Bert2LSTM
from KD_Lib.models import lenet, nin, shallow, lstm
from KD_Lib.models.resnet import resnet_book

import pandas as pd


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

#
#   MODEL TESTS
#

def test_resnet():
    params = [4, 4, 8, 8, 16]
    ResNet18(params)
    ResNet34(params)
    ResNet50(params)
    ResNet101(params)
    ResNet152(params)

def test_attention_model():
    params = [4, 4, 8, 8, 16]
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    model = ResNet152(params, att=True)
    sample_output = model(sample_input)
    print(sample_output)


def test_NIN():
    sample_input = torch.ones(size=(1, 1, 32, 32), requires_grad=False)
    model = nin.NetworkInNetwork(10, 1)
    sample_output = model(sample_input)
    print(sample_output)


def test_shallow():
    sample_input = torch.ones(size=(1, 1, 32, 32), requires_grad=False)
    model = shallow.Shallow(32)
    sample_output = model(sample_input)
    print(sample_output)


def test_lenet():
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    model = lenet.LeNet()
    sample_output = model(sample_input)
    print(sample_output)


def test_modlenet():
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    model = lenet.ModLeNet()
    sample_output = model(sample_input)
    print(sample_output)

def test_LSTMNet():
    sample_input = torch.tensor([[1,2,8,3,2],[2,4,99,1,7]])
    
    # Simple LSTM
    model = lstm.LSTMNet(num_classes=2,batch_size=2, dropout_prob=0.5)
    sample_output = model(sample_input)
    print(sample_output)

    # Bidirectional LSTM
    model = lstm.LSTMNet(num_classes=2,batch_size=2, dropout_prob=0.5)
    sample_output = model(sample_input)
    print(sample_output)


#
#   Strategy TESTS
#


def test_original():
    teac = shallow.Shallow(hidden_size=400)
    stud = shallow.Shallow(hidden_size=100)

    t_optimizer = optim.SGD(teac.parameters(), 0.01)
    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    orig = original(teac, stud, train_loader, test_loader, t_optimizer,
                    s_optimizer, loss='RKD', rkd_angle=0.4, rkd_dist=0.6)

    orig.train_teacher(epochs=0, plot_losses=False, save_model=False)
    orig.train_student(epochs=0, plot_losses=False, save_model=False)
    orig.evaluate(teacher=False)
    orig.get_parameters()


def test_TAKD():
    teacher = resnet_book['50']([4, 4, 8, 8, 16], num_channel=1)
    assistants = []
    temp = resnet_book['34']([4, 4, 8, 8, 16], num_channel=1)
    assistants.append(temp)
    temp = resnet_book['34']([4, 4, 8, 8, 16], num_channel=1)
    assistants.append(temp)

    student = resnet_book['18']([4, 4, 8, 8, 16], num_channel=1)

    teacher_optimizer = optim.Adam(teacher.parameters())
    assistant_optimizers = []
    assistant_optimizers.append(optim.Adam(assistants[0].parameters()))
    assistant_optimizers.append(optim.Adam(assistants[1].parameters()))
    student_optimizer = optim.Adam(student.parameters())

    assistant_train_order = [[-1], [-1, 0]]

    distil = TAKD(teacher, assistants, student, assistant_train_order,
                  train_loader, test_loader, teacher_optimizer,
                  assistant_optimizers, student_optimizer)

    distil.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distil.train_assistants(epochs=0, plot_losses=False, save_model=False)
    distil.train_student(epochs=0, plot_losses=False, save_model=False)
    distil.get_parameters()


def test_attention():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10, True)
    student_model = ResNet18(student_params, 1, 10, True)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    att = attention(teacher_model, student_model, train_loader, test_loader,
                    t_optimizer, s_optimizer, loss='ATTENTION')

    att.train_teacher(epochs=0, plot_losses=False, save_model=False)
    att.train_student(epochs=0, plot_losses=False, save_model=False)
    att.evaluate(teacher=False)
    att.get_parameters()


def test_NoisyTeacher():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    experiment = NoisyTeacher(teacher_model, student_model, train_loader,
                              test_loader, t_optimizer, s_optimizer,
                              alpha=0.4, noise_variance=0.2, device='cpu')

    experiment.train_teacher(epochs=0, plot_losses=False, save_model=False)
    experiment.train_student(epochs=0, plot_losses=False, save_model=False)
    experiment.evaluate(teacher=False)
    experiment.get_parameters()

def test_bert2lstm():
    data_csv = './KD_Lib/Bert2Lstm/IMDB Dataset.csv'
    df = pd.read_csv(data_csv)
    df['sentiment'].replace({'negative':0, 'positive':1},inplace=True)

    train_df = df.iloc[[0,1,2,-3,-2,-1],:]
    val_df = df.iloc[[10,15,20,-30,-20,-10],:]


    text_field, train_loader = get_essentials(train_df)

    student_model = lstm.LSTMNet(input_dim=len(text_field.vocab),num_classes=2,batch_size=2, dropout_prob=0.5)
    optimizer = torch.optim.Adam(student_model.parameters())
    
    experiment = Bert2LSTM(student_model,train_loader,None,optimizer,train_df,val_df)
    experiment.train_teacher(plot_losses=False, save_model=False)
    experiment.train_student(plot_losses=False, save_model=False)


