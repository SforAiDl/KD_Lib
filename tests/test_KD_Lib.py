# -*- coding: utf-8 -*-
"""Tests for `KD_Lib` package."""

import pytest
import random

from KD_Lib import KD_Lib
from KD_Lib.KD_Lib.original.mnist import mnist



@pytest.fixture


def test_mnist():
    mnist()


