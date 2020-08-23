import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from copy import deepcopy
import matplotlib.pyplot as plt

from .utils import add_noise
from KD_Lib.KD.common import BaseClass


class NoisyTeacher(BaseClass):
    """
    Implementation of Knowledge distillation using a noisy teacher from the paper "Deep
    Model Compression: Distilling Knowledge from Noisy Teachers" https://arxiv.org/pdf/1610.09650.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param alpha (float): Threshold for deciding if noise needs to be added
    :param noise_variance (float): Variance parameter for adding noise
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
        alpha=0.5,
        noise_variance=0.1,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(NoisyTeacher, self).__init__(
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

        self.alpha = alpha
        self.noise_variance = noise_variance

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        if random.uniform(0, 1) <= self.alpha:
            y_pred_teacher = add_noise(y_pred_teacher, self.noise_variance)

        loss = (1.0 - self.distil_weight) * F.cross_entropy(y_pred_student, y_true)
        loss += (self.distil_weight * self.temp * self.temp) * self.loss_fn(
            F.log_softmax(y_pred_student / self.temp, dim=1),
            F.softmax(y_pred_teacher / self.temp, dim=1),
        )
        return loss
