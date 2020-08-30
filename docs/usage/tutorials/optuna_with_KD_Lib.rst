=========================================
Hyperparameter Tuning using Optuna
=========================================

Hyperparameter optimization is one of the crucial steps in training machine learning models. It is often 
quite a tedious process with many parameters to optimize and long training times for models.
Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning
You can find more about Optuna `here <https://github.com/optuna/optuna>`_

Optuna an be installed using *pip* -

.. code-block:: console
    $ pip install optuna

or using *conda* -

.. code-block:: console
    $ conda install -c conda-forge optuna

To search for the best hyperparameters fot the VanillaKD algorithm -

.. code-block:: python

    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms
    from KD_Lib.KD import VanillaKD

    import optuna
    from sklearn.externals import joblib

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

    # Optuna requires defining an objective function 
    # The hyperparameters are then optimized for maximizing/minimizing this objective function
    
    def tune_VanillaKD(trial):

        teacher_model = <your model>
        student_model = <your model>

        # Define hyperparams and choose what ranges they should be trialled for

        lr = trial.suggest_float("lr", 1e-4, 1e-1)
        momentum = trial.suggest_float("momentum", 0.9, 0.99)
        optimizer = trial.suggest_categorical('optimizer',[optim.SGD, optim.Adam])

        teacher_optimizer = optimizer(teacher_model.parameters(), lr, momentum)
        student_optimizer = optimizer(student_model.parameters(), lr, momentum)

        temperature = trial.suggest_float("temperature", 5.0, 20.0)
        distil_weight = trial.suggest_float("distil_weight", 0.0, 1.0)

        loss_fn = trial.suggest_categorical('optimizer',[nn.KLDivLoss(), nn.MSELoss()])

        # Instiate disitller object using KD_Lib and train

        distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
                              teacher_optimizer, student_optimizer, tempereature, distil_weight, device)
        distiller.train_teacher(epochs=10)
        distiller.train_student(epochs=10)
        test_accuracy = disitller.evaluate()

        # The objective function must return the quantity we're trying to maximize/minimize

        return test_accuracy

    # Create a study

    study = optuna.create_study(study_name="Hyperparameter Optimization",
                                direction="maximize")
    study.optimize(tune_VanillaKD, n_trials=10)

    # Access results

    results = study.trials_dataframe()
    results.head()

    # Get best values of hyperparameter

    for key, value in study.best_trial.__dict__.items():
    print("{} : {}".format(key, value))
    
    # Write results of the study

    joblib.dump(study, <your path>)

    # Access results at a later time

    study = joblib.load(<your path>)
    results = study.trials_dataframe()
    results.head()
    




    
