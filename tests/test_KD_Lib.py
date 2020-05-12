# -*- coding: utf-8 -*-
"""Tests for `KD_Lib` package."""

import pytest
import random

from KD_Lib.TAKD.main import main_TAKD
from KD_Lib import KD_Lib


@pytest.fixture
def generate_numbers():
    """Sample pytest fixture. Generates list of random integers.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """

    return random.sample(range(100), 10)


def test_sum_numbers(generate_numbers):
    """Sample test function for sum_numbers, using pytest fixture."""

    our_result = KD_Lib.sum_numbers(generate_numbers)
    assert our_result == sum(generate_numbers)


def test_max_number(generate_numbers):
    """Sample test function for max_number, using pytest fixture."""

    our_result = KD_Lib.max_number(generate_numbers)
    assert our_result == max(generate_numbers)


def test_TAKD():
    config = {
        'teacher': {
            'name': 'resnet101',
            'params': [32, 32, 64, 64, 128],
            'optimizer': 'adam'
        },
        'assistants': [
            {
                'name': 'resnet50',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam'
            },
            {
                'name': 'resnet34',
                'params': [32, 32, 64, 64, 128],
                'optimizer': 'adam'
            },
        ],
        'student': {
            'name': 'resnet18',
            'params': [16, 32, 32, 16, 8],
            'optimizer': 'adam'
        },
        'dataset': {
            'name': 'cifar10',
            'location': './data/cifar10',
            'batch_size': 128
        },
        'loss_function': 'cross_entropy',
        'assistant_train_order': [[-1], [-1, 0]]

    }
    main_TAKD(config)
