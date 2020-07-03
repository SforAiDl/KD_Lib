import torch
from torch import nn
import torch.nn.functional as F

def PKTLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PKTLoss, self).__init__()
        self.eps = eps

    def forward(self, teacher, student):
        std_norm = torch.sqrt(torch.sum(student ** 2, dim=1, keepdim=True))
        student = student / (std_norm + self.eps)
        student[student != student] = 0

        t_norm = torch.sqrt(torch.sum(teacher ** 2, dim=1, keepdim=True))
        teacher = teacher / (t_norm + self.eps)
        teacher[teacher != teacher] = 0

        st_sim = (torch.mm(student, student.transpose(0, 1)) + 1) / 2
        t_sim = (torch.mm(teacher, teacher.transpose(0, 1)) + 1) / 2

        st_prob = st_sim / torch.sum(st_sim, dim=1, keepdim=True)
        t_prob = t_sim / torch.sum(t_tim, dim=1, keepdim=True)

        loss = torch.mean(t_prob * torch.log((t_prob + self.eps) / (st_prob + self.eps)))

        return loss


