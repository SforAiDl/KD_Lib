import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import glob
from copy import deepcopy

from KD_Lib.KD.common import BaseClass


class BANN(BaseClass):
    """
    Implementation of paper "Born Again Neural Networks"
    https://arxiv.org/abs/1805.04770

    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation
    :param optimizer (torch.optim.*): Optimizer for training
    :param num_gen (int): Number of generations to train.
    :param loss_fn (torch.nn.Module): Loss Function used for first model in gen.
                                      later, KLDivLoss is used.
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        num_gen,
        loss_fn=nn.CrossEntropyLoss(),
        epoch_interval=5,
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(BANN, self).__init__(
            student_model,
            student_model,
            train_loader,
            val_loader,
            optimizer,
            optimizer,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )
        self.init_weights = deepcopy(student_model.state_dict())
        self.init_optim = deepcopy(optimizer.state_dict())
        self.num_gen = num_gen
        self.gen = 0

    def train_student(
        self,
        epochs=10,
        plot_losses=False,
        save_model=True,
        save_model_pth="./models/student-{}.pth",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the student per generation
        :param plot_losses (bool): True if you want to plot the losses for every generation
        :param save_model (bool): True if you want to save the student model (Set true if you want to use models for later evaluation)
        :param save_model_pth (str): Path where you want to save the student model
        """
        try:
            fmt = save_model_pth.format(1)
        except:
            print("Invalid save_model_pth, allow {\} for generation number")
            return
        for k in range(self.num_gen):
            print("Born Again : Gen {}/{}".format(k + 1, self.num_gen))

            self._train_student(
                epochs, plot_losses, save_model, save_model_pth.format(k + 1)
            )

            # Use best model in k-1 gen as last model
            self.teacher_model.load_state_dict(self.best_student_model_weights)
            # Reset model for next generation
            self.student_model.load_state_dict(self.init_weights)
            # Reset optimizer for next generation
            self.optimizer_student.load_state_dict(self.init_optim)
            self.gen += 1

    def evaluate(self, models_dir="./models"):
        """
        Evaluate method for printing accuracies of the trained network

        :param models_dir (str): Location of stored models. (default: ./models)
        """
        print("Evaluating Model Ensemble")
        models_dir = glob.glob(os.path.join(models_dir, "*.pth"))
        len_models = len(models_dir)
        outputs = []
        model = self.student_model
        for model_weight in models_dir:
            model.load_state_dict(torch.load(model_weight))
            output, _ = self._evaluate_model(model, verbose=False)
            outputs.append(output)
        print("Total Models: ", len(outputs))
        total = len(self.val_loader)
        print("Total Samples: ", total)
        correct = 0
        for idx, (data, target) in enumerate(self.val_loader):
            target = target.to(self.device)
            output = outputs[0][idx] / len_models
            for k in range(1, len_models):
                output += outputs[k][idx] / len_models

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = {correct / total}
        print("-" * 80)
        print(f"Accuracy: {accuracy}")

        return accuracy

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """
        if self.gen == 0:
            return self.loss_fn(y_pred_student, y_true)

        s_i = F.log_softmax(y_pred_student / self.temp, dim=1)
        t_i = F.softmax(y_pred_teacher / self.temp, dim=1)
        KD_loss = nn.KLDivLoss()(s_i, t_i) * (
            self.distil_weight * self.temp * self.temp
        )
        KD_loss += F.cross_entropy(y_pred_student, y_true) * (1.0 - self.distil_weight)

        return KD_loss
