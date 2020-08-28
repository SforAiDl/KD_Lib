import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from copy import deepcopy

from KD_Lib.KD.common import BaseClass


class ProbShift(BaseClass):
    """
    Implementation of the knowledge adjustment technique from the paper
    "Preparing Lessons: Improve Knowledge Distillation with Better Supervision"
    https://arxiv.org/abs/1911.07471

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param method (string): Knowledge adjustment method used to correct the teacher's incorrect predictions. "LSR" takes additional prameter "correct_prob"
    :param correct_prob(float): The probability which is given to the correct class when "LSR" is chosen
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
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
        correct_prob=0.9,
        loss_fn=nn.KLDivLoss(),
        temp=20.0,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        super(ProbShift, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss_fn,
            temp,
            device,
            log,
            logdir,
        )

        self.correct_prob = correct_prob

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        start = 0
        count = 0

        num_classes = y_pred_teacher.shape[1]
        soft_pred_teacher = torch.Tensor([]).to(self.device)

        for i in range(y_pred_teacher.shape[0]):

            if torch.argmax(y_pred_teacher[i]) != y_true[i]:

                if i:
                    soft_pred_teacher = torch.cat(
                        (
                            soft_pred_teacher,
                            F.softmax(y_pred_teacher[start:i, :] / self.temp, dim=1),
                        ),
                        0,
                    )

                start = i + 1
                count += 1

                _, top_indices = torch.topk(y_pred_teacher[i], 2)
                index = torch.arange(num_classes).to(self.device)
                index[top_indices[0]] = top_indices[1]
                index[top_indices[1]] = top_indices[0]

                ps = torch.zeros_like(y_pred_teacher[i]).scatter_(
                    0, index, F.softmax(y_pred_teacher[i] / self.temp, dim=1)
                )
                soft_pred_teacher = torch.cat((soft_pred_teacher, ps.view(1, -1)), 0)

        if count:
            soft_pred_teacher = torch.cat(
                (
                    soft_pred_teacher,
                    F.softmax(y_pred_teacher[start:, :] / self.temp, dim=1),
                ),
                0,
            )
            loss = (self.temp * self.temp) * self.loss_fn(
                soft_pred_teacher, F.log_softmax(y_pred_student, dim=1)
            )

        else:
            loss = (self.temp * self.temp) * self.loss_fn(
                F.softmax(y_pred_teacher / self.temp, dim=1),
                F.log_softmax(y_pred_student, dim=1),
            )

        return loss
