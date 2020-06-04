import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from KD_Lib.common import BaseClass

import matplotlib.pyplot as plt
from copy import deepcopy


class BANN(BaseClass):
    """
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer (torch.optim.*): Optimizer used for training.
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param epoch_interval (int): Number of epochs after which teacher anchor points are created
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        num_gen,
        loss_fn=nn.CrossEntropyLoss(),
        epoch_interval=5,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(BANN, self).__init__(
            student_model,
            student_model,
            train_loader,
            val_loader,
            optimizer,
            optimizer,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )
        self.init_weights = deepcopy(student_model.state_dict())
        self.init_optim = deepcopy(optimizer)
        self.num_gen = num_gen
        self.gen = 0

    def train(self, epochs=10, plot_losses=False, save_model=True, save_model_pth="./models/student-{}.pth"):
        for k in range(self.num_gen):
            print("Born Again : Gen {}/{}".format(k+1, self.num_gen))

            self.train_student(epochs, plot_losses, save_model, save_model_pth.format(k+1))

            # Use best model in k-1 gen as last model
            self.teacher_model.load_state_dict(self.best_student_model_weights)
            # Reset model for next generation
            self.student_model.load_state_dict(self.init_weights)
            # Reset optimizer for next generation
            self.optimizer_student = deepcopy(self.init_optim)
            self.gen += 1


    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """
        if self.gen == 0:
            return self.loss_fn(y_pred_student, y_true)

        s_i = F.log_softmax(y_pred_student/self.temp, dim=1)
        t_i = F.softmax(y_pred_teacher/self.temp, dim=1)
        KD_loss = nn.KLDivLoss()(s_i,t_i) * self.distil_weight
        KD_loss += F.cross_entropy(y_pred_student, y_true) * (1. - self.distil_weight)

        return KD_loss

