KD_Lib
======


.. image:: https://travis-ci.com/SforAiDl/KD_Lib.svg?branch=master
    :target: https://travis-ci.com/SforAiDl/KD_Lib

A Pytorch Library to help extend all Knowledge Distillation works

Installation :
==============

==============
Stable release
==============
KD_Lib is compatible with Python 3.6 or later and also depends on pytorch. The easiest way to install KD_Lib is with pip, Python's preferred package installer.

``$ pip install KD-Lib``

Note that KD_Lib is an active project and routinely publishes new releases. In order to upgrade KD_Lib to the latest version, use pip as follows.

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

+-----------------------------------------------------------+----------------------------------+----------------------+
|  Paper                                                    |  Link                            | Repository (KD_Lib/) |
+===========================================================+==================================+======================+
| Distilling the Knowledge in a Neural Network              | https://arxiv.org/abs/1503.02531 | orig                 |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Improved Knowledge Distillation via Teacher Assistant     | https://arxiv.org/abs/1902.03393 | TAKD                 |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Relational Knowledge Distillation                         | https://arxiv.org/abs/1904.05068 | RKD                  |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Distilling Knowledge from Noisy Teachers                  | https://arxiv.org/abs/1610.09650 | noisy                |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Paying More Attention To The Attention                    | https://arxiv.org/abs/1612.03928 | attention            |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Revisit Knowledge Distillation: a Teacher-free Framework  | https://arxiv.org/abs/1909.11723 | teacher_free         |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Mean Teachers are Better Role Models                      | https://arxiv.org/abs/1703.01780 | mean_teacher         |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Knowledge Distillation via Route Constrained Optimization | https://arxiv.org/abs/1904.09149 | RCO                  |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Born Again Neural Networks                                | https://arxiv.org/abs/1805.04770 | BANN                 |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Preparing Lessons: Improve Knowledge Distillation with    | https://arxiv.org/abs/1911.07471 | KDBS                 |
| Better Supervision                                        |                                  |                      |
+-----------------------------------------------------------+----------------------------------+----------------------+