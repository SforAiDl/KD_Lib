import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from copy import deepcopy

from KD_Lib.KD.common import BaseClass


class CSKD(BaseClass):
    """
    Implementation of "Regularizing Class-wise Predictions via Self-knowledge Distillation"
     https://arxiv.org/pdf/2003.13964.pdf

    :param teacher_model (torch.nn.Module): Teacher model -> Should be None
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher -> Should be None
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module):  Calculates loss during distillation
    :param temp (float): Temperature parameter for distillation
    :param lambda (float): loss controlling parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=nn.CrossEntropyLoss(),
        temp=4.0,
        lamda=1,
        distil_weight=0.4,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):
        super(CSKD, self).__init__(
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
        self.lamda = lamda
        if teacher_model is not None or optimizer_teacher is not None:
            print(
                "Error!!! Teacher model and Teacher optimizer should be None for self-distillation, please refer to the documentation."
            )
        assert teacher_model == None

    def calculate_kd_loss(self, y_pred_pair_1, y_pred_pair_2):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_pair_1 (torch.FloatTensor): Prediction made by the student model for first pair elements
        :param y_pred_pair_2 (torch.FloatTensor): Prediction made by the student models for second pair elements
        """
        log_p = torch.log_softmax(y_pred_pair_1 / self.temp, dim=1)
        q = torch.softmax(y_pred_pair_2 / self.temp, dim=1)
        loss = (
            nn.KLDivLoss(reduction="sum")(log_p, q)
            * (self.temp ** 2)
            / y_pred_pair_1.size(0)
        )

        return loss

    def train_distil_model(
        self,
        model,
        optimizer,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pth",
    ):
        """
        Function used for distillation

        :param teacher(list or tuple): Teachers used for distillation
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

        for epoch in range(epochs):
            correct = 0
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            train_loss = 0
            total = 0
            train_cls_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                batch_size = data.size(0)

                targets_ = target[: batch_size // 2]
                outputs = model(data[: batch_size // 2])
                loss = torch.mean(self.loss_fn(outputs, targets_))
                train_loss += loss.item()

                with torch.no_grad():
                    outputs_cls = model(data[batch_size // 2 :])
                cls_loss = self.calculate_kd_loss(outputs, outputs_cls.detach())
                loss += self.lamda * cls_loss
                train_cls_loss += cls_loss.item()

                _, pred = torch.max(outputs, 1)
                total += targets_.size(0)
                correct += pred.eq(targets_.data).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset
            epoch_val_acc = self.evaluate(teacher=False)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_model_weights = deepcopy(model.state_dict())

            if self.log:
                self.writer.add_scalar(
                    "Training loss/Student", epoch_loss / batch_idx, epochs
                )
                self.write.add_scalar(
                    "Training Cls loss/Student", train_cls_loss / batch_idx, epochs
                )
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
                self.writer.add_scalar(
                    "Validation accuracy/Student", epoch_val_acc, epochs
                )

            loss_arr.append(epoch_loss)

            print(
                f"Epoch: {epoch+1}, Loss: {epoch_loss/batch_idx} Loss_cls: {train_cls_loss/batch_idx}, Accuracy: {epoch_acc*100.}"
            )

        model.load_state_dict(self.best_model_weights)
        if save_model:
            torch.save(model.state_dict(), save_model_pth)

        if plot_losses:
            plt.plot(loss_arr)

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
            self.student_model,
            self.optimizer_student,
            epochs,
            plot_losses,
            save_model,
            save_model_pth,
        )
