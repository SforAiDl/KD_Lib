import torch
from torch import nn
import torch.nn.functional as F


def pairwaise_distance(output):
    """
    Function for calculating pairwise distance

    :param output (torch.FloatTensor): Input for calculating pairwise distance
    """

    output_squared = output.pow(2).sum(dim=1)
    product = torch.mm(output, output.t())
    result = output_squared.unsqueeze(0) + output_squared.unsqueeze(1) - 2 * product
    result[range(len(output)), range(len(output))] = 0
    return result.sqrt()


class RKDDistanceLoss(nn.Module):
    """
    Module for calculating RKD Distance Loss
    """

    def forward(self, teacher, student, normalize=False):
        """
        Forward function

        :param teacher (torch.FloatTensor): Prediction made by the teacher model
        :param student (torch.FloatTensor): Prediction made by the student model
        :param normalize (bool): True if inputs need to be normalized
        """

        with torch.no_grad():
            t = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            if normalize:
                t = F.normalize(t, p=2, dim=2)
            t = torch.bmm(t, t.transpose(1, 2)).view(-1)

        s = student.unsqueeze(0) - student.unsqueeze(1)
        if normalize:
            s = F.normalize(s, p=2, dim=2)
        s = torch.bmm(s, s.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s, t, reduction="mean")


class RKDAngleLoss(nn.Module):
    """
    Module for calculating RKD Angle Loss
    """

    def forward(self, teacher, student, normalize=False):
        """
        Forward function

        :param teacher (torch.FloatTensor): Prediction made by the teacher model
        :param student (torch.FloatTensor): Prediction made by the student model
        :param normalize (bool): True if inputs need to be normalized
        """

        with torch.no_grad():
            t = pairwaise_distance(teacher)
            if normalize:
                t = F.normalize(t, p=2, dim=2)

        s = pairwaise_distance(student)
        if normalize:
            s = F.normalize(s, p=2, dim=2)
        return F.smooth_l1_loss(s, t, reduction="mean")


angle_loss = RKDAngleLoss()
distance_loss = RKDDistanceLoss()


class RKDLoss(nn.Module):
    """
    Module for calculating RKD Distance Loss

    :param dist_ratio (float): Distance ratio for RKD loss if used
    :param angle_ratio (float): Angle ratio for RKD loss if used
    """

    def __init__(self, dist_ratio=0.5, angle_ratio=0.5):
        super(RKDLoss, self).__init__()
        self.dist_ratio = dist_ratio
        self.angle_ratio = angle_ratio

    def forward(self, teacher, student, normalize=False):
        """
        Forward function

        :param teacher (torch.FloatTensor): Prediction made by the teacher model
        :param student (torch.FloatTensor): Prediction made by the student model
        :param normalize (bool): True if inputs need to be normalized
        """

        loss = angle_loss(teacher, student, normalize) * self.angle_ratio
        loss += distance_loss(teacher, student, normalize) * self.dist_ratio
        return loss
