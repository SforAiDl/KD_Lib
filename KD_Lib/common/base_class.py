import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from copy import deepcopy


class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.loss_fn = loss_fn
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        try:
            torch.tensor(0).to(device)
            self.device = device
        except:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = "cpu"

    def train_teacher(
        self,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/teacher.pt",
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        self.teacher_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        print("Training Teacher... ")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.teacher_model(data)

                if isinstance(out, tuple):
                    out = out[0]

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = F.cross_entropy(out, label)

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(f"Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

            self.post_epoch_call(ep)

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pth",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        print("\nTraining student...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)

                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(f"Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def evaluate(self, teacher=False):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model)
        else:
            model = deepcopy(self.student_model)
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print("-" * 80)
        print(f"Accuracy: {correct/length_of_dataset}")

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print(f"Total parameters for the teacher network are: {teacher_params}")
        print(f"Total parameters for the student network are: {student_params}")

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass
