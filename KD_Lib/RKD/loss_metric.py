import torch
from torch import nn
import torch.nn.functional as F


def pairwaise_distance(output):
    output_squared = output.pow(2).sum(dim=1)
    product = torch.mm(output, output.t())
    result = (output_squared.unsqueeze(0) + output_squared.unsqueeze(1)
              - 2 * product)
    result[range(len(output)), range(len(output))] = 0
    return result


class RKDDistanceLoss(nn.Module):
    def forward(self, teacher, student, normalize=False):
        with torch.no_grad():
            t = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            if normalize:
                t = F.normalize(t, p=2, dim=2)
            t = torch.bmm(t, t.transpose(1, 2)).view(-1)

        s = student.unsqueeze(0) - student.unsqueeze(1)
        if normalize:
            s = F.normalize(s, p=2, dim=2)
        s = torch.bmm(s, s.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s, t, reduction='elementwise_mean')


class RKDAngleLoss(nn.Module):
    def forward(self, teacher, student, normalize=False):
        with torch.no_grad():
            t = pairwaise_distance(teacher)
            if normalize:
                t = F.normalize(t, p=2, dim=2)

        s = pairwaise_distance(student)
        if normalize:
            s = F.normalize(s, p=2, dim=2)

        return F.smooth_l1_loss(s, t, reduction='elementwise_mean')
