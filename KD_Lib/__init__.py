# -*- coding: utf-8 -*-

"""Top-level package for KD_Lib."""

__author__ = """Het Shah"""
__email__ = 'divhet163@gmail.com'
__version__ = '0.0.1'

from .models import resnet, shallow, nin, lenet, lstm
from .original import original
from .attention import attention
from .TAKD import TAKD
from .Bert2Lstm import bert2lstm, utils
