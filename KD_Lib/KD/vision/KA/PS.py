import torch
import torch.nn as nn
import torch.nn.functional as F

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
        loss_fn=nn.KLDivLoss(reduction="batchmean"),
        temp=20.0,
        ka_weight=0.5,
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
            loss_fn=loss_fn,
            temp=temp,
            distil_weight=ka_weight,
            device=device,
            log=log,
            logdir=logdir,
        )
        self.label_loss = nn.CrossEntropyLoss().to(self.device)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Swaps probabilty of incorrectly predicted class with true class.

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """
        soft_pred_student = F.softmax(y_pred_student / self.temp, dim=1)

        with torch.no_grad():
            soft_pred_teacher = F.softmax(y_pred_teacher / self.temp, dim=1)
            for i in range(y_pred_teacher.shape[0]):
                t_label = torch.argmax(soft_pred_teacher[i])
                if t_label != y_true[i]:
                    temp_prob = soft_pred_teacher[i][t_label]
                    soft_pred_teacher[i][t_label] = soft_pred_teacher[i][y_true[i]]
                    soft_pred_teacher[i][y_true[i]] = temp_prob

        ka_loss = (
            self.temp * self.temp * self.loss_fn(soft_pred_teacher, soft_pred_student)
        )
        ce_loss = self.temp * self.label_loss(y_pred_student / self.temp, y_true)
        return ka_loss * self.distil_weight + (1 - self.distil_weight) * ce_loss
