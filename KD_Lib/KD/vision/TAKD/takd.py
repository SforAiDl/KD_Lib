import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from copy import deepcopy

from KD_Lib.KD.common import BaseClass


class TAKD(BaseClass):
    """
    Implementation of assisted Knowledge distillation from the paper "Improved Knowledge
    Distillation via Teacher Assistant" https://arxiv.org/pdf/1902.03393.pdf

    :param teacher_model (torch.nn.Module): Teacher model
    :param assistant_models (list or tuple): Assistant models
    :param student_model (torch.nn.Module): Student model
    :param assistant_train_order (list or tuple or array): Order of training for that assistant
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_assistants (torch.optim.*): Optimizer used for training assistants
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        assistant_models,
        student_model,
        assistant_train_order,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_assistants,
        optimizer_student,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distil_weight=0.4,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(TAKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )
        self.assistant_models = assistant_models
        self.optimizer_assistants = optimizer_assistants
        self.assistant_train_order = assistant_train_order
        self.log_softmax = nn.LogSoftmax(dim=1).to(self.device)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        loss = (1 - self.distil_weight) * self.ce_fn(y_pred_student, y_true)
        loss += (self.distil_weight * self.temp * self.temp) * self.loss_fn(
            self.log_softmax(y_pred_student / self.temp),
            self.log_softmax(y_pred_teacher / self.temp),
        )

        return loss

    def train_distil_model(
        self,
        teachers,
        model,
        optimizer,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pt",
    ):
        """
        Function used for distillation

        :param teachers (list or tuple): Teachers used for distillation
        :param model (nn.Module): Model learning during distillation
        :param optimizer (torch.optim.*): Optimizer used for training
        :param epochs (int): Number of epochs to train
        :param plot_losses (bool): True if the loss curves need to be plotted
        :param save_model (bool): True if the model needs to be saved
        :param save_model_path (str): Path used for storing the model
        """

        model.train()
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_model_weights = deepcopy(model.state_dict())

        if not isinstance(teachers, list):
            teachers = [teachers]

        for epoch in range(epochs):
            correct = 0
            epoch_loss = 0.0
            for _, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                scores = model(data)

                pred = scores.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                teacher_output = teachers[0](data)
                for i in range(1, len(teachers)):
                    teacher_output += teachers[i](data)
                teacher_output /= len(teachers)

                loss = self.calculate_kd_loss(scores, teacher_output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset

            _, epoch_val_acc = self._evaluate_model(model)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_model_weights = deepcopy(model.state_dict())

            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

        model.load_state_dict(self.best_model_weights)
        if save_model:
            torch.save(model.state_dict(), save_model_pth)

    def train_assistants(
        self, epochs=20, plot_losses=True, save_model=True, save_dir="./models/"
    ):
        """
        Function used for training assistants

        :param epochs (int): Number of epochs to train
        :param plot_losses (bool): True if the loss curves need to be plotted
        :param save_model (bool): True if the model needs to be saved
        :param save_dir (str): Path used for storing the trained asssistant models
        """

        count = 0
        for assistant in self.assistant_models:
            trainers = []
            train_order = self.assistant_train_order[count]
            for elem in train_order:
                if elem == -1:
                    trainers.append(self.teacher_model)

                else:
                    trainers.append(self.assistant_models[elem])

            self.train_distil_model(
                trainers,
                assistant,
                self.optimizer_assistants[count],
                epochs,
                plot_losses,
                save_model,
                save_dir + "assistant_" + str(count) + ".pt",
            )
            count += 1

    def train_student(
        self,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pth",
    ):
        """
        Function used for distilling knowledge to student

        :param plot_losses (bool): True if the loss curves need to be plotted
        :param save_model (bool): True if the model needs to be saved
        :param save_model_path (str): Path used for storing the trained student model
        """

        self.train_distil_model(
            self.assistant_models,
            self.student_model,
            self.optimizer_student,
            epochs,
            plot_losses,
            save_model,
            save_model_pth,
        )
