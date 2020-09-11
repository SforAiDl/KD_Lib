import torch
from torch import nn
import torch.nn.functional as F
from KD_Lib.KD.common import BaseClass


def symmetric_mse_loss(input1, input2):
    return torch.sum((input1 - input2) ** 2)


class MeanTeacher(BaseClass):
    """
    Implementation of Knowledge distillation using a mean teacher from the
    paper "Mean teachers are better role models:Weight-averaged consistency
    targets improvesemi-supervised deep learning results"
    https://arxiv.org/abs/1703.01780

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation
    :param optimizer_teacher (torch.optim.*): Optimizer for training teacher
    :param optimizer_student (torch.optim.*): Optimizer for training student

    :param loss (str): Consistency criterion for loss
    :param class_loss (torch.nn.Module): Class Criterion for loss
    :param res_loss (torch.nn.Module): Residual Logit Criterion for loss

    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device for training; 'cpu' for cpu and 'cuda' for gpu
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
        class_loss=nn.CrossEntropyLoss(),
        res_loss=symmetric_mse_loss,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(MeanTeacher, self).__init__(
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
        self.class_loss = class_loss.to(self.device)
        try:
            self.res_loss = res_loss.to(self.device)
        except:
            self.res_loss = res_loss
        self.loss_fn = loss_fn.to(self.device)
        self.log_softmax = nn.LogSoftmax(dim=1).to(self.device)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """
        class_logit, consis_logit = y_pred_student
        class_loss = self.class_loss(class_logit, y_true)

        num_classes = consis_logit.size()[1]
        res_loss = self.res_loss(class_logit, consis_logit) / num_classes

        student_softmax = self.log_softmax(consis_logit, dim=1)
        teacher_softmax = self.log_softmax(y_pred_teacher[0], dim=1)
        consis_loss = self.loss_fn(student_softmax, teacher_softmax) / num_classes

        return class_loss + res_loss + consis_loss

    def post_epoch_call(self, epoch):
        """
        Exponentially updates the weights of teacher model.

        :param epoch (int): current epoch
        """
        alpha = min(1e-3, epoch / (epoch + 1))
        param_zip = zip(
            self.teacher_model.parameters(), self.student_model.parameters()
        )
        for teacher_param, param in param_zip:
            teacher_param.data.mul_(alpha).add_(1 - alpha, param.data)
