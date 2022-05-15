from copy import deepcopy

import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from KD_Lib.KD import (
    BANN,
    DML,
    RCO,
    TAKD,
    Attention,
    BaseClass,
    LabelSmoothReg,
    MeanTeacher,
    MessyCollab,
    NoisyTeacher,
    ProbShift,
    SelfTraining,
    SoftRandom,
    VanillaKD,
    VirtualTeacher,
)
from KD_Lib.KD.text.BERT2LSTM import BERT2LSTM
from KD_Lib.KD.text.BERT2LSTM.utils import get_essentials

from .utils import MockImageClassifier, MockVisionDataset

img_size = (32, 32)
img_channels = 3
n_classes = 10
len_dataset = 4
batch_size = 2

train_loader = test_loader = DataLoader(
    MockVisionDataset(
        size=img_size, n_classes=n_classes, length=len_dataset, n_channel=img_channels
    ),
    batch_size=batch_size,
)

mock_vision_model = MockImageClassifier(
    size=img_size, n_classes=n_classes, n_channels=img_channels
)


## BERT to LSTM data

# data_csv = "./KD_Lib/KD/text/BERT2LSTM/IMDB_Dataset.csv"
# df = pd.read_csv(data_csv)
# df["sentiment"].replace({"negative": 0, "positive": 1}, inplace=True)

# train_df = df.iloc[:6, :]
# val_df = df.iloc[6:, :]

# text_field, bert2lstm_train_loader = get_essentials(train_df)


def test_VanillaKD():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = VanillaKD(
        teacher, student, train_loader, test_loader, t_optimizer, s_optimizer, log=True
    )

    distiller.train_teacher(epochs=1, plot_losses=True, save_model=True)
    distiller.train_student(epochs=1, plot_losses=True, save_model=True)
    distiller.evaluate(teacher=False)
    distiller.get_parameters()


def test_TAKD():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    assistants = [deepcopy(mock_vision_model) for _ in range(2)]

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

    distil.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distil.train_assistants(epochs=1, plot_losses=False, save_model=False)
    distil.train_student(epochs=1, plot_losses=False, save_model=False)
    distil.get_parameters()


def test_attention():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    att = Attention(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    att.train_teacher(epochs=1, plot_losses=False, save_model=False)
    att.train_student(epochs=1, plot_losses=False, save_model=False)
    att.evaluate(teacher=False)
    att.get_parameters()


def test_NoisyTeacher():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    experiment = NoisyTeacher(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
        alpha=0.4,
        noise_variance=0.2,
        device="cpu",
    )

    experiment.train_teacher(epochs=1, plot_losses=False, save_model=False)
    experiment.train_student(epochs=1, plot_losses=False, save_model=False)
    experiment.evaluate(teacher=False)
    experiment.get_parameters()


def test_VirtualTeacher():

    student = deepcopy(mock_vision_model)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = VirtualTeacher(student, train_loader, test_loader, s_optimizer)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_SelfTraining():

    student = deepcopy(mock_vision_model)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = SelfTraining(student, train_loader, test_loader, s_optimizer)

    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


# def test_mean_teacher():
#     teacher_params = [16, 16, 32, 16, 16]
#     student_params = [16, 16, 16, 16, 16]
#     teacher_model = ResNet50(teacher_params, 1, 10, mean=True)
#     student_model = ResNet18(student_params, 1, 10, mean=True)

#     t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
#     s_optimizer = optim.SGD(student_model.parameters(), 0.01)

#     mt = MeanTeacher(
#         teacher_model,
#         student_model,
#         train_loader,
#         test_loader,
#         t_optimizer,
#         s_optimizer,
#     )

#     mt.train_teacher(epochs=1, plot_losses=False, save_model=False)
#     mt.train_student(epochs=1, plot_losses=False, save_model=False)
#     mt.evaluate()
#     mt.get_parameters()


def test_RCO():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = RCO(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


# def test_BANN():
#     params = [4, 4, 4, 4, 4]
#     model = ResNet50(params, 1, 10)
#     optimizer = optim.SGD(model.parameters(), 0.01)

#     distiller = BANN(model, train_loader, test_loader, optimizer, num_gen=2)

#     distiller.train_student(epochs=1, plot_losses=False, save_model=False)
#     distiller.evaluate()


def test_PS():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = ProbShift(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_LSR():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = LabelSmoothReg(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_soft_random():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = SoftRandom(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


def test_messy_collab():

    teacher = deepcopy(mock_vision_model)
    student = deepcopy(mock_vision_model)

    t_optimizer = optim.SGD(teacher.parameters(), 0.01)
    s_optimizer = optim.SGD(student.parameters(), 0.01)

    distiller = MessyCollab(
        teacher,
        student,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()


# def test_bert2lstm():
#     student_model = LSTMNet(
#         input_dim=len(text_field.vocab), num_classes=2, dropout_prob=0.5
#     )
#     optimizer = optim.Adam(student_model.parameters())
#
#     experiment = BERT2LSTM(
#         student_model, bert2lstm_train_loader, bert2lstm_train_loader, optimizer, train_df, val_df
#     )
#     # experiment.train_teacher(epochs=1, plot_losses=False, save_model=False)
#     experiment.train_student(epochs=1, plot_losses=False, save_model=False)
#     experiment.evaluate_student()
#     experiment.evaluate_teacher()


def test_DML():

    student_1 = deepcopy(mock_vision_model)
    student_2 = deepcopy(mock_vision_model)

    student_cohort = (student_1, student_2)

    s_optimizer_1 = optim.SGD(student_1.parameters(), 0.01)
    s_optimizer_2 = optim.SGD(student_2.parameters(), 0.01)

    student_optimizers = (s_optimizer_1, s_optimizer_2)

    distiller = DML(
        student_cohort,
        train_loader,
        test_loader,
        student_optimizers,
        log=True,
        logdir=".",
    )

    distiller.train_students(
        epochs=1, plot_losses=True, save_model=True, save_model_path="./student.pt"
    )
    distiller.evaluate()
    distiller.get_parameters()
