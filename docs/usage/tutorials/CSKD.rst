======================
Regularizing Class-wise Predictions via Self-knowledge Distillation using KD_Lib
======================

To implement the most basic version of knowledge distillation from `Regularizing Class-wise Predictions via Self-knowledge Distillation <https://arxiv.org/abs/2003.13964>`_
and plot losses

.. code-block:: python

    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms
    from KD_Lib.KD.vision import CSKD
    from KD_Lib.KD.vision.CSKD.sampler import load_dataset


    # Define datasets, dataloaders, models and optimizers
    
    # Note that the dataloader should sample according to pairwise sampling. (Refer to KD_Lib -> KD -> vision -> CSKD -> sampler.py)
    train_loader, val_loader = load_dataset('cifar100', '~/data/', 'pair', batch_size=128)


    student_model = <your model>

    student_optimizer = optim.SGD(student_model.parameters(), 0.01)

    # Now, this is where KD_Lib comes into the picture

    distiller = CSKD(None, student_model, train_loader, test_loader, 
                          None, student_optimizer)  
    distiller.train_student(epochs=5, plot_losses=True, save_model=True)    # Train the student network
    distiller.evaluate(teacher=False)                                       # Evaluate the student network
    distiller.get_parameters()                                              # A utility function to get the number of parameters in the teacher and the student network 
