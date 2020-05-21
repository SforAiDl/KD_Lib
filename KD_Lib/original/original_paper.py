import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .model import teacher, student
from KD_Lib.RKD import RKDLoss
from KD_Lib.common import BaseClass

class original(BaseClass):
    def __init__(self, teacher_model, student_model, train_loader, val_loader, optimizer_teacher, optimizer_student, loss='MSE', temp=20.0, distil_weight=0.5, device='cpu', **kwargs):
        super(original, self).__init__(
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
        if self.loss.upper() == 'MSE':
            self.loss_fn = nn.MSELoss()

        elif self.loss_fn.upper() == 'KL':
            self.loss_fn = nn.KLDivLoss()

        elif self.loss_fn.upper() == 'RKD':
            self.loss_fn = RKDLoss(dist_ratio=rkd_dist, angle_ratio=rkd_angle)  

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        soft_teacher_out = F.softmax(y_pred_teacher/self.temp)
        soft_student_out = F.softmax(y_pred_student/self.temp)

        loss = (1-self.distl_weight) * F.cross_entropy(soft_student_out, y_true)
        loss += self.distl_weight * self.loss_fn(soft_teacher_out, soft_student_out)

        return loss 