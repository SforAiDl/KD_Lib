======================
VanillaKD using KD_Lib
======================

To implement the most basic version of knowledge distillation from `Distilling the Knowledge in a Neural Network <https://arxiv.org/abs/1503.02531>`_
and plot losses

.. code-block:: python

    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms
    from KD_Lib.KD import VanillaKD

    # Define datasets, dataloaders, models and optimizers

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
    distiller.get_parameters()                                              # A utility function to get the number of parameters in the teacher and the student network 
