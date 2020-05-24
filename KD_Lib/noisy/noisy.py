import random
import torch
from torch.nn import MSELoss
from KD_Lib.common import BaseClass


def add_noise(x, variance=0.1):
    return x * (1 + (variance**0.5) * torch.randn_like(x))


class NoisyTeacher(BaseClass):
    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 optimizer_teacher, optimizer_student,
                 loss_fn=MSELoss(), alpha=0.5, noise_variance=0.1,
                 loss='MSE', temp=20.0, distil_weight=0.5, device='cpu', 
                 log=False, logdir='./Experiments'):
        super(NoisyTeacher, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss,
            temp,
            distil_weight,
            device,
            log,
            logdir
        )

        self.loss_fn = loss_fn
        self.alpha = alpha
        self.noise_variance = noise_variance

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        if random.uniform(0, 1) <= self.alpha:
            y_pred_teacher = add_noise(y_pred_teacher, self.noise_variance)
        return self.loss_fn(y_pred_student, y_pred_teacher)
