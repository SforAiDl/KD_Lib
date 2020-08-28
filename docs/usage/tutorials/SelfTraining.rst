===========================================
Self Training using KD_Lib
===========================================

`Paper <https://arxiv.org/abs/1909.11723>`_

* The student model is first trained in the normal way to obtain a pre-trained model, which is then used as the teacher to 
train itself by transferring soft targets


To use the self training algorithm to train a student on MNIST for 5 epcohs -

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from KD_Lib.KD import SelfTraining

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

    # Set device to be trained on

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define student and teacher models

    student_model = <your model>

    # Define optimizers

    student_optimizer = optim.SGD(student_model.parameters(), lr=0.01)


    # Train using KD_Lib

    distiller = SelfTraining(student_model, train_loader, test_loader, student_optimizer, 
                             device=device)  
    distiller.train_student(epochs=5)                                      # Train the student model
    distiller.evaluate()                                                    # Evaluate the student model
    

