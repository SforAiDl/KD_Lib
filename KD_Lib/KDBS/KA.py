import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from KD_Lib.common import BaseClass

import matplotlib.pyplot as plt
from copy import deepcopy


class KnowledgeAdjustment(BaseClass):
    """
    Implementation of the knowledge adjustment technique from the paper
    from the paper "Knowledge Distillation via Route Constrained Optimization" 
    https://arxiv.org/abs/1904.09149

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
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
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        method="PS",
        correct_prob=0.9,
        loss_fn=nn.KLDivLoss(),
        temp=20.0,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        super(KnowledgeAdjustment, self).__init__(
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

        self.method = method.upper()
        self.correct_prob = correct_prob


    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        num_channels = y_pred_teacher.shape[1]
        soft_teacher_pred = F.softmax(y_pred_teacher / self.temp, dim=1)

        if self.method == 'LSR':

          for i in range(soft_teacher_pred.shape[0]):

            if torch.argmax(soft_teacher_pred[i]) != y_true[i]:
              soft_teacher_pred[i] = torch.ones_like(y_pred_student).to(self.device)
              soft_teacher_pred[i] *=  (1 - self.correct_prob) / (num_channels - 1)
              soft_label[i, y_true[i]] = self.correct_prob

        elif self.method == "PS":

          for i in range(soft_teacher_pred.shape[0]):

            _, top_indices = torch.topk(soft_teacher_pred[i], 2)
            if top_indices[0] != y_true[i]:
              soft_teacher_pred[i, top_indices[0]], soft_teacher_pred[i, top_indices[1]] = soft_teacher_pred[i, top_indices[1]], soft_teacher_pred[i, top_indices[0]]

        loss = self.loss_fn(F.log_softmax(y_pred_student, dim=1),
                            soft_teacher_pred
                            )
        
        return loss
        






