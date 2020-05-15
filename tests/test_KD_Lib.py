# -*- coding: utf-8 -*-
"""Tests for `KD_Lib` package."""

#import pytest
#import random

from KD_Lib.TAKD.main import main_TAKD
from KD_Lib import KD_Lib
from KD_Lib.original.mnist import mnist


def test_mnist():
    mnist(epochs=0)


def test_TAKD():
    config = {
        'teacher': {
            'name': 'resnet101',
            'params': [32, 32, 64, 64, 128],
            'optimizer': 'adam',
            'train_epoch': 0
        },
        'assistants': [
            {
                'name': 'resnet50',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam',
                'train_epoch': 0
            },
            {
                'name': 'resnet34',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam',
                'train_epoch': 0
            },
        ],
        'student': {
            'name': 'resnet18',
            'params': [16, 32, 32, 16, 8],
            'optimizer': 'adam',
            'train_epoch': 0
        },
        'dataset': {
            'name': 'mnist',
            'location': './data/mnist',
            'batch_size': 128,
            'num_classes': 10,
            'num_channels': 1
        },
        'loss_function': 'cross_entropy',
        'assistant_train_order': [[-1], [-1, 0]]
    }
    main_TAKD(config)
