import torch.nn as nn
import torch.nn.functional as F
from KD_Lib.RKD import RKDLoss
from KD_Lib.common import BaseClass


class original(BaseClass):
    def __init__(self, teacher_model, student_model, train_loader, val_loader,
                 optimizer_teacher, optimizer_student, loss='MSE', temp=20.0,
                 distil_weight=0.5, device='cpu', rkd_angle=None, rkd_dist=None, 
                 log=False, logdir='./Experiments'):
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
            device,
            log,
            logdir
        )
        if self.loss.upper() == 'MSE':
            self.loss_fn = nn.MSELoss()

        elif self.loss.upper() == 'KL':
            self.loss_fn = nn.KLDivLoss()

        elif self.loss.upper() == 'RKD':
            self.loss_fn = RKDLoss(dist_ratio=rkd_dist, angle_ratio=rkd_angle)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        soft_teacher_out = F.softmax(y_pred_teacher/self.temp, dim=0)
        soft_student_out = F.softmax(y_pred_student/self.temp, dim=0)

        loss = (1-self.distil_weight) * F.cross_entropy(soft_student_out,
                                                       y_true)
        loss += self.distil_weight * self.loss_fn(soft_teacher_out,
                                                 soft_student_out)
        return loss
