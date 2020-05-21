import random
from KD_Lib.common import BaseClass
from .utils import add_noise
from torch.nn import MSELoss


class NoisyTeacher(BaseClass):
    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 optimizer_teacher, optimizer_student,
                 loss_fn=MSELoss(), alpha=0.5, noise_variance=0.1,
                 loss='MSE', temp=20.0, distil_weight=0.5, device='cpu'):
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
            device
        )

        self.loss_fn = loss_fn
        self.alpha = alpha
        self.noise_variance = noise_variance

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        if random.uniform(0, 1) <= self.alpha:
            y_pred_teacher = add_noise(y_pred_teacher, self.noise_variance)
        return self.loss_fn(y_pred_student, y_pred_teacher)
