KD_Lib
======


.. image:: https://img.shields.io/travis/SforAiDl/KD_Lib.svg
        :target: https://travis-ci.org/SforAiDl/KD_Lib

A Pytorch Library to help extend all Knowledge Distillation works

Installation :
==============

==============
Stable release
==============
KD_Lib is compatible with Python 3.6 or later and also depends on pytorch. The easiest way to install KD_Lib is with pip, Python's preferred package installer.

``$ pip install KD-Lib``

Note that GenRL is an active project and routinely publishes new releases. In order to upgrade GenRL to the latest version, use pip as follows.

``$ pip install -U KD-Lib``

=================
Build from source
=================

If you intend to install the latest unreleased version of the library (i.e from source), you can simply do:

| ``$ git clone https://github.com/SforAiDl/KD_Lib.git``
| ``$ cd KD_lib``
| ``$ python setup.py install``

Currently implemented works
===========================

| Original MNIST Paper: https://arxiv.org/abs/1503.02531 
| Improved Knowledge Distillation via Teacher Assistant: https://arxiv.org/abs/1902.03393
| Relational Knowledge Distillation: https://arxiv.org/abs/1904.05068
| Distilling Knowledge from Noisy Teachers: https://arxiv.org/pdf/1610.09650.pdf
| Paying More Attention To The Attention - Improving the Performance of CNNs via Attention Transfer: https://arxiv.org/pdf/1612.03928.pdf
