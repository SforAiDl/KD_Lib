import torch.nn.functional as F
from torch import nn


class ATLoss(nn.Module):
    def __init__(self, norm_type=2):
        super(ATLoss, self).__init__()
        self.p = norm_type

    def forward(self, teacher_output, student_output):
        A_t = teacher_output[1:]
        A_s = student_output[1:]
        loss = 0.0
        for (layerT, layerS) in zip(A_t, A_s):
            xT = self.single_at_loss(layerT)
            xS = self.single_at_loss(layerS)
            loss += (xS - xT).pow(self.p).mean()
        return loss

    def single_at_loss(self, activation):
        return F.normalize(
            activation.pow(self.p).mean(1).view(activation.size(0), -1))
