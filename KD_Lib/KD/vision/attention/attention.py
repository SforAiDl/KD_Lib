import torch.nn.functional as F

from KD_Lib.KD.common import BaseClass
from .loss_metric import ATLoss


class Attention(BaseClass):
    """
    Implementation of attention-based Knowledge distillation from the paper "Paying More
    Attention To The Attention - Improving the Performance of CNNs via Attention Transfer"
    https://arxiv.org/pdf/1612.03928.pdf

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
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(Attention, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            ATLoss(),
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )

        self.loss_fn = self.loss_fn.to(self.device)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        loss = (
            (1.0 - self.distil_weight)
            * self.temp
            * F.cross_entropy(y_pred_student[0] / self.temp, y_true)
        )
        loss += self.distil_weight * self.loss_fn(y_pred_teacher, y_pred_student)
        return loss
