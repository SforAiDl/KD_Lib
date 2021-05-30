import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from copy import deepcopy


class SelfTraining:
    """
    Implementation of the self training kowledge distillation framework from the paper
    "Revisit Knowledge Distillation: a Teacher-free Framework" https://arxiv.org/abs/1909.11723


    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param rkd_angle (float): Angle ratio for RKD loss if used
    :param rkd_dist (float): Distance ratio for RKD loss if used
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        student_model,
        train_loader,
        val_loader,
        optimizer_student,
        loss_fn=nn.KLDivLoss(),
        temp=10.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_student = optimizer_student
        self.loss_fn = loss_fn
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        try:
            torch.Tensor(0).to(device)
            self.device = device
        except:
            print(
                "Either an invalid device or CUDA is not available. Defaulting to CPU."
            )
            self.device = "cpu"

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

        self_teacher = deepcopy(self.student_model)
        optimizer_self_teacher = optim.SGD(self_teacher.parameters(), 0.01, 0.9)
        self_teacher.train()

        length_of_dataset = len(self.train_loader.dataset)

        print("\nTraining self teacher...")

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.to(self.device)

                out = self_teacher(data)

                loss = F.cross_entropy(out, label)

                if isinstance(out, tuple):
                    out = out[0]

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                optimizer_self_teacher.zero_grad()
                loss.backward()
                optimizer_self_teacher.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset

            if self.log:
                self.writer.add_scalar("Training loss/Self Teacher", epoch_loss, epochs)
                self.writer.add_scalar(
                    "Training accuracy/Self Teacher", epoch_acc, epochs
                )

            print(f"Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

        self_teacher.eval()
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
                self_teacher_out = self_teacher(data)

                loss = self.calculate_kd_loss(student_out, self_teacher_out, label)

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset

            epoch_val_acc = self.evaluate()

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
                self.writer.add_scalar(
                    "Validation accuracy/Student", epoch_val_acc, epochs
                )

            loss_arr.append(epoch_loss)
            print(f"Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        loss = (1 - self.distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (self.distil_weight) * self.loss_fn(
            F.log_softmax(y_pred_student, dim=1),
            F.softmax(y_pred_teacher / self.temp, dim=1),
        )
        return loss

    def evaluate(self):
        """
        Evaluate method for printing accuracies of the trained network

        """

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

        accuracy = correct / length_of_dataset
        print("-" * 80)
        print(f"Accuracy: {accuracy}")
        return accuracy

    def get_parameters(self):
        """(
        Get the number of parameters for the student network
        """

        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print(f"Total parameters for the student network are: {student_params}")
