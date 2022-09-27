<h1 align="center">KD-Lib</h1>
<h3 align="center">A PyTorch model compression library containing easy-to-use methods for knowledge distillation, pruning, and quantization</h3>

<div align='center'>

[![Downloads](https://pepy.tech/badge/kd-lib)](https://pepy.tech/project/kd-lib)
[![Tests](https://github.com/SforAiDl/KD_Lib/actions/workflows/python-package-test.yml/badge.svg)](https://github.com/SforAiDl/KD_Lib/actions/workflows/python-package-test.yml)
[![Docs](https://readthedocs.org/projects/kd-lib/badge/?version=latest)](https://kd-lib.readthedocs.io/en/latest/?badge=latest)

**[Documentation](https://kd-lib.readthedocs.io/en/latest/)** | **[Tutorials](https://kd-lib.readthedocs.io/en/latest/usage/tutorials/index.html)**

</div>

## Installation

### From source (recommended)

```shell

https://github.com/SforAiDl/KD_Lib.git
cd KD_Lib
python setup.py install

```

### From PyPI

```shell

pip install KD-Lib

```

## Example usage

To implement the most basic version of knowledge distillation from [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) and plot loss curves:

```python

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import VanillaKD

# This part is where you define your datasets, dataloaders, models and optimizers

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

teacher_model = <your model>
student_model = <your model>

teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
student_optimizer = optim.SGD(student_model.parameters(), 0.01)

# Now, this is where KD_Lib comes into the picture

distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader, 
                      teacher_optimizer, student_optimizer)  
distiller.train_teacher(epochs=5, plot_losses=True, save_model=True)    # Train the teacher network
distiller.train_student(epochs=5, plot_losses=True, save_model=True)    # Train the student network
distiller.evaluate(teacher=False)                                       # Evaluate the student network
distiller.get_parameters()                                              # A utility function to get the number of 
                                                                        # parameters in the  teacher and the student network

```

To train a collection of 3 models in an online fashion using the framework in [Deep Mutual Learning](https://arxiv.org/abs/1706.00384)
and log training details to Tensorboard: 

```python

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import DML
from KD_Lib.models import ResNet18, ResNet50          # To use models packaged in KD_Lib

# Define your datasets, dataloaders, models and optimizers

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
    shuffle=True,
)

student_params = [4, 4, 4, 4, 4]
student_model_1 = ResNet50(student_params, 1, 10)
student_model_2 = ResNet18(student_params, 1, 10)

student_cohort = [student_model_1, student_model_2]

student_optimizer_1 = optim.SGD(student_model_1.parameters(), 0.01)
student_optimizer_2 = optim.SGD(student_model_2.parameters(), 0.01)

student_optimizers = [student_optimizer_1, student_optimizer_2]

# Now, this is where KD_Lib comes into the picture 

distiller = DML(student_cohort, train_loader, test_loader, student_optimizers, log=True, logdir="./logs")

distiller.train_students(epochs=5)
distiller.evaluate()
distiller.get_parameters()

```

## Methods Implemented

Some benchmark results can be found in the [logs](./logs.rst) file.

|  Paper / Method                                           |  Link                            | Repository (KD_Lib/) |
| ----------------------------------------------------------|----------------------------------|----------------------|
| Distilling the Knowledge in a Neural Network              | https://arxiv.org/abs/1503.02531 | KD/vision/vanilla    |
| Improved Knowledge Distillation via Teacher Assistant     | https://arxiv.org/abs/1902.03393 | KD/vision/TAKD       |
| Relational Knowledge Distillation                         | https://arxiv.org/abs/1904.05068 | KD/vision/RKD        |
| Distilling Knowledge from Noisy Teachers                  | https://arxiv.org/abs/1610.09650 | KD/vision/noisy      |
| Paying More Attention To The Attention                    | https://arxiv.org/abs/1612.03928 | KD/vision/attention  |
| Revisit Knowledge Distillation: a Teacher-free <br> Framework  | https://arxiv.org/abs/1909.11723 |KD/vision/teacher_free|
| Mean Teachers are Better Role Models                      | https://arxiv.org/abs/1703.01780 |KD/vision/mean_teacher|
| Knowledge Distillation via Route Constrained <br> Optimization | https://arxiv.org/abs/1904.09149 | KD/vision/RCO        |
| Born Again Neural Networks                                | https://arxiv.org/abs/1805.04770 | KD/vision/BANN       |
| Preparing Lessons: Improve Knowledge Distillation <br> with Better Supervision | https://arxiv.org/abs/1911.07471 | KD/vision/KA |
| Improving Generalization Robustness with Noisy <br> Collaboration in Knowledge Distillation | https://arxiv.org/abs/1910.05057 | KD/vision/noisy|
| Distilling Task-Specific Knowledge from BERT into <br> Simple Neural Networks | https://arxiv.org/abs/1903.12136 | KD/text/BERT2LSTM |
| Deep Mutual Learning                                      | https://arxiv.org/abs/1706.00384 | KD/vision/DML        |
| The Lottery Ticket Hypothesis: Finding Sparse, <br> Trainable Neural Networks | https://arxiv.org/abs/1803.03635 | Pruning/lottery_tickets|
| Regularizing Class-wise Predictions via <br> Self-knowledge Distillation | https://arxiv.org/abs/2003.13964 | KD/vision/CSDK |

<br>

Please cite our pre-print if you find `KD-Lib` useful in any way :)

```bibtex

@misc{shah2020kdlib,
  title={KD-Lib: A PyTorch library for Knowledge Distillation, Pruning and Quantization}, 
  author={Het Shah and Avishree Khare and Neelay Shah and Khizir Siddiqui},
  year={2020},
  eprint={2011.14691},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

```
