import pandas as pd
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from KD_Lib.KD import (
    TAKD,
    Attention,
    VanillaKD,
    VirtualTeacher,
    SelfTraining,
    NoisyTeacher,
    SoftRandom,
    MessyCollab,
    MeanTeacher,
    RCO,
    BANN,
    ProbShift,
    LabelSmoothReg,
    DML,
    BaseClass,
)

from KD_Lib.KD.text.BERT2LSTM.utils import get_essentials
from KD_Lib.KD.text.BERT2LSTM import BERT2LSTM

from KD_Lib.models import ResNet18, ResNet50, Shallow, resnet_book


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=4,
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
    batch_size=4,
    shuffle=True,
)

## BERT to LSTM data

# data_csv = "./KD_Lib/KD/text/BERT2LSTM/IMDB_Dataset.csv"
# df = pd.read_csv(data_csv)
# df["sentiment"].replace({"negative": 0, "positive": 1}, inplace=True)

# train_df = df.iloc[:6, :]
# val_df = df.iloc[6:, :]

# text_field, bert2lstm_train_loader = get_essentials(train_df)


def test_VanillaKD():
    teac = Shallow(hidden_size=400)
    stud = Shallow(hidden_size=100)

    t_optimizer = optim.SGD(teac.parameters(), 0.01)
    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    distiller = VanillaKD(
        teac, stud, train_loader, test_loader, t_optimizer, s_optimizer, log=True
    )

    distiller.train_teacher(epochs=1, plot_losses=True, save_model=True)
    distiller.train_student(epochs=1, plot_losses=True, save_model=True)
    distiller.evaluate(teacher=False)
    distiller.get_parameters()

    del teac, stud, distiller, t_optimizer, s_optimizer


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

    distil.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distil.train_assistants(epochs=1, plot_losses=False, save_model=False)
    distil.train_student(epochs=1, plot_losses=False, save_model=False)
    distil.get_parameters()

    del (
        teacher,
        assistants,
        student,
        distil,
        teacher_optimizer,
        assistant_optimizers,
        student_optimizer,
    )


def test_attention():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10, True)
    student_model = ResNet18(student_params, 1, 10, True)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    att = Attention(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    att.train_teacher(epochs=1, plot_losses=False, save_model=False)
    att.train_student(epochs=1, plot_losses=False, save_model=False)
    att.evaluate(teacher=False)
    att.get_parameters()

    del teacher_model, student_model, att, t_optimizer, s_optimizer


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

    experiment.train_teacher(epochs=1, plot_losses=False, save_model=False)
    experiment.train_student(epochs=1, plot_losses=False, save_model=False)
    experiment.evaluate(teacher=False)
    experiment.get_parameters()

    del teacher_model, student_model, experiment, t_optimizer, s_optimizer


def test_VirtualTeacher():
    stud = Shallow(hidden_size=300)

    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    distiller = VirtualTeacher(stud, train_loader, test_loader, s_optimizer)

    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del stud, distiller, s_optimizer


def test_SelfTraining():
    stud = Shallow(hidden_size=300)

    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    distiller = SelfTraining(stud, train_loader, test_loader, s_optimizer)

    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del stud, distiller, s_optimizer


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

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del teacher_model, student_model, distiller, t_optimizer, s_optimizer


# def test_BANN():
#     params = [4, 4, 4, 4, 4]
#     model = ResNet50(params, 1, 10)
#     optimizer = optim.SGD(model.parameters(), 0.01)

#     distiller = BANN(model, train_loader, test_loader, optimizer, num_gen=2)

#     distiller.train_student(epochs=1, plot_losses=False, save_model=False)
#     distiller.evaluate()


def test_PS():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = ProbShift(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del teacher_model, student_model, distiller, t_optimizer, s_optimizer


def test_LSR():
    teacher_params = [4, 4, 8, 4, 4]
    student_params = [4, 4, 4, 4, 4]
    teacher_model = ResNet50(teacher_params, 1, 10)
    student_model = ResNet18(student_params, 1, 10)

    t_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    s_optimizer = optim.SGD(student_model.parameters(), 0.01)

    distiller = LabelSmoothReg(
        teacher_model,
        student_model,
        train_loader,
        test_loader,
        t_optimizer,
        s_optimizer,
    )

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del teacher_model, student_model, distiller, t_optimizer, s_optimizer


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

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del teacher_model, student_model, distiller, t_optimizer, s_optimizer


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

    distiller.train_teacher(epochs=1, plot_losses=False, save_model=False)
    distiller.train_student(epochs=1, plot_losses=False, save_model=False)
    distiller.evaluate()
    distiller.get_parameters()

    del teacher_model, student_model, distiller, t_optimizer, s_optimizer


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

    student_params = [4, 4, 4, 4, 4]
    student_model_1 = ResNet50(student_params, 1, 10)
    student_model_2 = ResNet18(student_params, 1, 10)

    student_cohort = (student_model_1, student_model_2)

    s_optimizer_1 = optim.SGD(student_model_1.parameters(), 0.01)
    s_optimizer_2 = optim.SGD(student_model_2.parameters(), 0.01)

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

    del student_model_1, student_model_2, distiller, s_optimizer_1, s_optimizer_2
