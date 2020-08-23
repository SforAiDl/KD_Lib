# -*- coding: utf-8 -*-

"""Top-level package for KD_Lib."""

__author__ = """Het Shah"""
__email__ = "divhet163@gmail.com"
__version__ = "0.0.3"

from .KD import (
    VanillaKD,
    VirtualTeacher,
    SelfTraining,
    TAKD,
    RKDLoss,
    RCO,
    NoisyTeacher,
    SoftRandom,
    MessyCollab,
    MeanTeacher,
    LabelSmoothReg,
    ProbShift,
    DML,
    BANN,
    Attention,
)

from .models import (
    ResNet,
    ResNet101,
    ResNet152,
    ResNet18,
    ResNet34,
    ResNet50,
    resnet_book,
    LSTMNet,
    Shallow,
    NetworkInNetwork,
    LeNet,
    ModLeNet,
)

from .Pruning import Lottery_Tickets_Pruner
