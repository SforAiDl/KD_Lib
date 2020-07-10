# -*- coding: utf-8 -*-
"""Tests for `KD_Lib` package."""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from KD_Lib.TAKD.takd import TAKD
from KD_Lib.attention.attention import attention
from KD_Lib.vanilla.vanilla_kd import VanillaKD
from KD_Lib.teacher_free.virtual_teacher import VirtualTeacher
from KD_Lib.teacher_free.self_training import SelfTraining
from KD_Lib.noisy.noisy_teacher import NoisyTeacher
from KD_Lib.noisy.soft_random import SoftRandom
from KD_Lib.noisy.messy_collab import MessyCollab
from KD_Lib.mean_teacher import MeanTeacher
from KD_Lib.RCO import RCO
from KD_Lib.BANN import BANN
from KD_Lib.KA import KnowledgeAdjustment
from KD_Lib.noisy import NoisyTeacher
from KD_Lib.Bert2Lstm.utils import get_essentials
from KD_Lib.Bert2Lstm.bert2lstm import Bert2LSTM
from KD_Lib.DML import DML
from KD_Lib.models import lenet, nin, shallow, lstm
from KD_Lib.models.resnet import resnet_book

from KD_Lib.Pruning.lottery_tickets import Lottery_Tickets_Pruner

import pandas as pd


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

## BERT to LSTM data
data_csv = "./KD_Lib/Bert2Lstm/IMDB_Dataset.csv"
df = pd.read_csv(data_csv)
df["sentiment"].replace({"negative": 0, "positive": 1}, inplace=True)

train_df = df.iloc[:6, :]
val_df = df.iloc[6:, :]

text_field, train_loader = get_essentials(train_df)


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


def test_meanteacher_model():
    params = [4, 4, 8, 8, 16]
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    model = ResNet152(params, mean=True)
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
    sample_input = torch.tensor([[1, 2, 8, 3, 2], [2, 4, 99, 1, 7]])

    # Simple LSTM
    model = lstm.LSTMNet(num_classes=2, batch_size=2, dropout_prob=0.5)
    sample_output = model(sample_input)
    print(sample_output)

    # Bidirectional LSTM
    model = lstm.LSTMNet(num_classes=2, batch_size=2, dropout_prob=0.5)
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

    distiller = VanillaKD(
        teac, stud, train_loader, test_loader, t_optimizer, s_optimizer
    )

    distiller.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate(teacher=False)
    distiller.get_parameters()


def test_TAKD():
    teacher = resnet_book["50"]([4, 4, 8, 8, 16], num_channel=1)
    assistants = []
    temp = resnet_book["34"]([4, 4, 8, 8, 16], num_channel=1)
    assistants.append(temp)
    temp = resnet_book["34"]([4, 4, 8, 8, 16], num_channel=1)
    assistants.append(temp)

    student = resnet_book["18"]([4, 4, 8, 8, 16], num_channel=1)

    teacher_optimizer = optim.Adam(teacher.parameters())
    assistant_optimizers = []
    assistant_optimizers.append(optim.Adam(assistants[0].parameters()))
    assistant_optimizers.append(optim.Adam(assistants[1].parameters()))
    student_optimizer = optim.Adam(student.parameters())

    assistant_train_order = [[-1], [-1, 0]]

    distil = TAKD(
        teacher,
        assistants,
        student,
        assistant_train_order,
        train_loader,
        test_loader,
        teacher_optimizer,
        assistant_optimizers,
        student_optimizer,
    )

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

    att = attention(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

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

    experiment = NoisyTeacher(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
        alpha=0.4,
        noise_variance=0.2,
        device="cpu",
    )

    experiment.train_teacher(epochs=0, plot_losses=False, save_model=False)
    experiment.train_student(epochs=0, plot_losses=False, save_model=False)
    experiment.evaluate(teacher=False)
    experiment.get_parameters()


def test_VirtualTeacher():
    stud = shallow.Shallow(hidden_size=300)

    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    distiller = VirtualTeacher(stud, train_loader, test_loader, s_optimizer)

    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_SelfTraining():
    stud = shallow.Shallow(hidden_size=300)

    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    distiller = SelfTraining(stud, train_loader, test_loader, s_optimizer)

    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_mean_teacher():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10, mean=True)
    student_model = ResNet18(student_params, 1, 10, mean=True)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    mt = MeanTeacher(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    mt.train_teacher(epochs=0, plot_losses=False, save_model=False)
    mt.train_student(epochs=0, plot_losses=False, save_model=False)
    mt.evaluate()
    mt.get_parameters()


def test_RCO():

    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = RCO(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_BANN():
    params = [4, 4, 4, 4, 4]
    model = ResNet50(params, 1, 10)
    optimizer = optim.SGD(model.parameters(), 0.01)

    distiller = BANN(model, train_loader, test_loader, optimizer, num_gen=2)

    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    # distiller.evaluate()


def test_KA_PS():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = KnowledgeAdjustment(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
        "PS",
    )

    distiller.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_KA_LSR():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = KnowledgeAdjustment(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
        "LSR",
    )

    distiller.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_soft_random():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = SoftRandom(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_messy_collab():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = MessyCollab(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=0, plot_losses=False, save_model=False)
    distiller.train_student(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_bert2lstm():
    student_model = lstm.LSTMNet(
        input_dim=len(text_field.vocab), num_classes=2, batch_size=2, dropout_prob=0.5
    )
    optimizer = torch.optim.Adam(student_model.parameters())

    experiment = Bert2LSTM(
        student_model, train_loader, train_loader, optimizer, train_df, val_df
    )
    # experiment.train_teacher(epochs=0, plot_losses=False, save_model=False)
    experiment.train_student(epochs=0, plot_losses=False, save_model=False)
    experiment.evaluate_student()
    experiment.evaluate_teacher()


def test_DML():

    student_params = [4, 4, 4, 4, 4]
    student_model_1 = ResNet50(student_params, 1, 10)
    student_model_2 = ResNet18(student_params, 1, 10)

    student_cohort = (student_model_1, student_model_2)

    s_optimizer_1 = optim.SGD(student_model_1.parameters(), 0.01)
    s_optimizer_2 = optim.SGD(student_model_2.parameters(), 0.01)

    student_optimizers = (s_optimizer_1, s_optimizer_2)

    distiller = DML(student_cohort, train_loader, test_loader, student_optimizers)

    distiller.train_students(epochs=0, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


#
# Pruning tests
#


def test_lottery_tickets():
    teacher_params = [4, 4, 8, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10, True)
    pruner = Lottery_Tickets_Pruner(teacher_model, train_loader, test_loader)
    pruner.prune(num_iterations=0, train_iterations=0, valid_freq=1, print_freq=1)
