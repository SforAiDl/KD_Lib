KD_Lib
======


.. image:: https://travis-ci.com/SforAiDl/KD_Lib.svg?branch=master
    :target: https://travis-ci.com/SforAiDl/KD_Lib

A PyTorch library to easily facilitate knowledge distillation for custom deep learning models

Installation :
==============

==============
Stable release
==============
KD_Lib is compatible with Python 3.6 or later and also depends on pytorch. The easiest way to install KD_Lib is with pip, Python's preferred package installer.

.. code-block:: console

    $ pip install KD-Lib

Note that KD_Lib is an active project and routinely publishes new releases. In order to upgrade KD_Lib to the latest version, use pip as follows.

.. code-block:: console

    $ pip install -U KD-Lib

=================
Build from source
=================

If you intend to install the latest unreleased version of the library (i.e from source), you can simply do:

.. code-block:: console

    $ git clone https://github.com/SforAiDl/KD_Lib.git
    $ cd KD_lib
    $ python setup.py install


Usage
======

To implement the most basic version of knowledge distillation from `Distilling the Knowledge in a Neural Network <https://arxiv.org/abs/1503.02531>`_
and plot losses

.. code-block:: console

    from KD_Lib.KD.vision.vanilla import VanillaKD

    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                          teacher_optimizer, student_optimizer)
    distiller.train_teacher(epochs=5, plot_losses=True, save_model=True)
    distiller.train_student(epochs=5, plot_losses=True, save_model=True)
    distiller.evaluate()


To train a collection of 3 models in an online fashion using the framework in `Deep Mutual Learning <https://arxiv.org/abs/1706.00384>`_
and log training details to Tensorboard

.. code-block:: console

    from KD_Lib.KD.vision.DML import DML

    student_cohort = (student_model_1, student_model_2, student_model_3)
    student_optimizers = (s_optimizer_1, s_optimizer_2, student_optimizer_3)
    
    distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, log=True)
    distiller.train_students(epochs=5)
    distiller.evaluate()


Currently implemented works
===========================

+-----------------------------------------------------------+----------------------------------+----------------------+
|  Paper                                                    |  Link                            | Repository (KD_Lib/) |
+===========================================================+==================================+======================+
| Distilling the Knowledge in a Neural Network              | https://arxiv.org/abs/1503.02531 | KD/vision/vanilla    |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Improved Knowledge Distillation via Teacher Assistant     | https://arxiv.org/abs/1902.03393 | KD/vision/TAKD       |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Relational Knowledge Distillation                         | https://arxiv.org/abs/1904.05068 | KD/vision/RKD        |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Distilling Knowledge from Noisy Teachers                  | https://arxiv.org/abs/1610.09650 | KD/vision/noisy      |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Paying More Attention To The Attention                    | https://arxiv.org/abs/1612.03928 | KD/vision/attention  |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Revisit Knowledge Distillation: a Teacher-free Framework  | https://arxiv.org/abs/1909.11723 |KD/vision/teacher_free|
+-----------------------------------------------------------+----------------------------------+----------------------+
| Mean Teachers are Better Role Models                      | https://arxiv.org/abs/1703.01780 |KD/vision/mean_teacher|
+-----------------------------------------------------------+----------------------------------+----------------------+
| Knowledge Distillation via Route Constrained Optimization | https://arxiv.org/abs/1904.09149 | KD/vision/RCO        |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Born Again Neural Networks                                | https://arxiv.org/abs/1805.04770 | KD/vision/BANN       |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Preparing Lessons: Improve Knowledge Distillation with    | https://arxiv.org/abs/1911.07471 | KD/vision/KA         |
| Better Supervision                                        |                                  |                      |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Improving Generalization Robustness with Noisy            | https://arxiv.org/abs/1910.05057 | KD/vision/noisy      |
| Collaboration in Knowledge Distillation                   |                                  |                      |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Distilling Task-Specific Knowledge from BERT into         | https://arxiv.org/abs/1903.12136 | KD/text/BERT2LSTM    |
| Simple Neural Networks                                    |                                  |                      |
+-----------------------------------------------------------+----------------------------------+----------------------+
| Deep Mutual Learning                                      | https://arxiv.org/abs/1706.00384 | KD/vision/DML        |
+-----------------------------------------------------------+----------------------------------+----------------------+
| The Lottery Ticket Hypothesis: Finding                    | https://arxiv.org/abs/1803.03635 | Pruning/             |
| Sparse, Trainable Neural Networks                         |                                  | lottery_tickets      |
+-----------------------------------------------------------+----------------------------------+----------------------+