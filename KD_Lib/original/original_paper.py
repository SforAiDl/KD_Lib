import torch.nn as nn
import torch.nn.functional as F
from KD_Lib.RKD import RKDLoss
from KD_Lib.common import BaseClass


class original(BaseClass):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the 
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss (str): Loss used for training
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param rkd_angle (float): Angle ratio for RKD loss if used
    :param rkd_dist (float): Distance ratio for RKD loss if used
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """
    
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
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model 
        :param y_true (torch.FloatTensor): Original label
        """
        soft_teacher_out = F.softmax(y_pred_teacher/self.temp, dim=0)
        soft_student_out = F.softmax(y_pred_student/self.temp, dim=0)

        loss = (1-self.distil_weight) * F.cross_entropy(soft_student_out,
                                                       y_true)
        loss += self.distil_weight * self.loss_fn(soft_teacher_out,
                                                 soft_student_out)
        return loss
