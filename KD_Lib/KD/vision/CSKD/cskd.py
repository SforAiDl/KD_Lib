import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from copy import deepcopy

from KD_Lib.KD.common import BaseClass


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)
        q = torch.softmax(target / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q) * (self.temp_factor ** 2) / input.size(0)
        return loss


class CSKD(BaseClass):
    """
    Implementation of assisted Knowledge distillation from the paper "Improved Knowledge
    Distillation via Teacher Assistant" https://arxiv.org/pdf/1902.03393.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
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
        temp=4.0,
        lamda=1,
        distil_weight=0.4,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(CSKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )
        self.lamda = lamda

    def train_distil_model(
        self,
        model,
        optimizer,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pth",
    ):
        """
        Function used for distillation

        :param teacher(list or tuple): Teachers used for distillation
        :param model (nn.Module): Model learning during distillation
        :param optimizer (torch.optim.*): Optimizer used for training
        :param epochs (int): Number of epochs to train
        :param plot_losses (bool): True if the loss curves need to be plotted
        :param save_model (bool): True if the model needs to be saved
        :param save_model_path (str): Path used for storing the model
        """

        kdloss = KDLoss(self.temp)
        model.train()
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_model_weights = deepcopy(model.state_dict())

        for epoch in range(epochs):
            correct = 0
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            train_loss = 0
            total = 0
            train_cls_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                batch_size = data.size(0)

                targets_ = target[: batch_size // 2]
                outputs = model(data[: batch_size // 2])
                loss = torch.mean(self.loss_fn(outputs, targets_))
                train_loss += loss.item()

                with torch.no_grad():
                    outputs_cls = model(data[batch_size // 2 :])
                cls_loss = kdloss(outputs, outputs_cls.detach())
                loss += self.lamda * cls_loss
                train_cls_loss += cls_loss.item()

                _, pred = torch.max(outputs, 1)
                # pred = outputs.argmax(dim=1, keepdim=True)
                total += targets_.size(0)
                correct += pred.eq(targets_.data).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_model_weights = deepcopy(model.state_dict())

            print(
                f"Epoch: {epoch}, Loss: {epoch_loss/batch_idx} Loss_cls: {train_cls_loss/batch_idx}, Accuracy: {epoch_acc*100.}"
            )

        model.load_state_dict(self.best_model_weights)
        if save_model:
            torch.save(model.state_dict(), save_model_pth)

    def train_student(
        self,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pth",
    ):
        """
        Function used for distilling knowledge to student

        :param plot_losses (bool): True if the loss curves need to be plotted
        :param save_model (bool): True if the model needs to be saved
        :param save_model_path (str): Path used for storing the trained student model
        """

        self.train_distil_model(
            self.student_model,
            self.optimizer_student,
            epochs,
            plot_losses,
            save_model,
            save_model_pth,
        )
