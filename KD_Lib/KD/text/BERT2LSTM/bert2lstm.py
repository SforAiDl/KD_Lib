import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from transformers import BertForSequenceClassification, AdamW, BertTokenizer

from KD_Lib.KD.common import BaseClass
from KD_Lib.KD.text.utils import get_bert_dataloader


class BERT2LSTM(BaseClass):
    """
    Implementation of Knowledge distillation from the paper "Distilling Task-Specific
    Knowledge from BERT into Simple Neural Networks" https://arxiv.org/pdf/1903.12136.pdf

    :param student_model (torch.nn.Module): Student model
    :param distill_train_loader (torch.utils.data.DataLoader): Student Training Dataloader for distillation
    :param distill_val_loader (torch.utils.data.DataLoader): Student Testing/validation Dataloader
    :param train_df (pandas.DataFrame): Dataframe for training the teacher model
    :param val_df (pandas.DataFrame): Dataframe for validating the teacher model
    :param loss_fn (torch.nn.module): Loss function
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        student_model,
        distill_train_loader,
        distill_val_loader,
        optimizer_student,
        train_df,
        val_df,
        num_classes=2,
        seed=42,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
        max_seq_length=128,
    ):

        teacher_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )

        optimizer_teacher = AdamW(teacher_model.parameters(), lr=2e-5, eps=1e-8)

        super(BERT2LSTM, self).__init__(
            teacher_model,
            student_model,
            distill_train_loader,
            distill_val_loader,
            optimizer_teacher,
            optimizer_student,
            None,
            None,
            distil_weight,
            device,
            log,
            logdir,
        )

        self.set_seed(42)

        self.train_df, self.val_df = train_df, val_df

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.max_seq_length = max_seq_length

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _get_teacher_dataloaders(self, batch_size=16, mode="train"):
        """
        Helper function for generating dataloaders for the teacher
        """
        df = self.val_df if (mode == "validate") else self.train_df

        return get_bert_dataloader(
            df, self.bert_tokenizer, self.max_seq_length, batch_size, mode
        )

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
        :param y_true (torch.FloatTensor): Original label
        """

        teacher_out = y_pred_teacher
        student_out = y_pred_student
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()

        loss = (1 - self.distil_weight) * self.criterion_ce(student_out, y_true)
        loss += (self.distil_weight) * self.criterion_mse(teacher_out, student_out)
        return loss

    def train_teacher(
        self,
        epochs=1,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/teacher.pt",
        train_batch_size=16,
        batch_print_freq=40,
        val_batch_size=16,
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        :param train_batch_size (int): Batch size paramter for generating dataloaders
        :param batch_print_freq (int): Frequency at which batch number needs to be printed per epoch
        """
        self.teacher_train_loader = self._get_teacher_dataloaders(
            train_batch_size, mode="train"
        )

        self.teacher_model.to(self.device)
        self.teacher_model.train()

        # training_stats = []
        loss_arr = []

        length_of_dataset = len(self.teacher_train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        print("Training Teacher... ")

        for ep in range(0, epochs):
            print("")
            print("======== Epoch {:} / {:} ========".format(ep + 1, epochs))

            epoch_loss = 0.0
            correct = 0

            for step, batch in enumerate(self.teacher_train_loader):
                if step % (batch_print_freq) == 0 and not step == 0:
                    print(
                        "  Batch {:>5,}  of  {:>5,}.".format(
                            step, len(self.teacher_train_loader)
                        )
                    )

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.optimizer_teacher.zero_grad()

                loss, logits = self.teacher_model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                epoch_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()
                preds = np.argmax(logits, axis=1).flatten()
                labels = label_ids.flatten()
                correct += np.sum(preds == labels)

                loss.backward()

                # For preventing exploding gradients
                torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), 1.0)

                self.optimizer_teacher.step()

            epoch_acc = correct / length_of_dataset
            print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

            _, epoch_val_acc = self.evaluate_teacher(val_batch_size)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)
                self.writer.add_scalar(
                    "Validation accuracy/Teacher", epoch_val_acc, epochs
                )

            loss_arr.append(epoch_loss)

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)

        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)

        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pth",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """

        self.teacher_distill_loader = self._get_teacher_dataloaders(
            batch_size=self.train_loader.batch_size, mode="distill"
        )

        y_pred_teacher = []

        print("Obtaining teacher predictions...")
        self.teacher_model.eval()
        self.teacher_model.to(self.device)

        for batch in self.teacher_distill_loader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                (loss, logits) = self.teacher_model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                logits = logits.detach().cpu().numpy()

                y_pred_teacher.append(logits)

        self.student_model.train()

        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())
        self.student_model.to(self.device)

        print("\nTraining student...")

        for ep in range(epochs):
            print("")
            print("======== Epoch {:} / {:} ========".format(ep + 1, epochs))

            epoch_loss = 0.0
            correct = 0

            for (data, data_len, label), bert_prob in zip(
                self.train_loader, y_pred_teacher
            ):
                data = data.to(self.device)
                data_len = data_len.to(self.device)
                label = label.to(self.device)

                bert_prob = torch.tensor(bert_prob, dtype=torch.float)
                teacher_out = bert_prob.to(self.device)

                self.optimizer_student.zero_grad()

                student_out = self.student_model(data, data_len).squeeze(1)

                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss.backward()

                # ##For preventing exploding gradients
                # torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)

                self.optimizer_student.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset
            print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

            _, epoch_val_acc = self.evaluate_student()
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
                self.writer.add_scalar(
                    "Validation accuracy/Student", epoch_val_acc, epochs
                )

            loss_arr.append(epoch_loss)
            print(f"Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

        self.student_model.load_state_dict(self.best_student_model_weights)

        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)

        if plot_losses:
            plt.plot(loss_arr)

    def evaluate_student(self, verbose=True):
        """
        Function used for evaluating student

        :param verbose (bool): True if the accuracy needs to be printed else False
        """

        self.student_model.eval()
        self.student_model.to(self.device)
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, data_len, target in self.val_loader:
                data = data.to(self.device)
                data_len = data_len.to(self.device)
                target = target.to(self.device)
                output = self.student_model(data, data_len).squeeze(1)
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset
        if verbose:
            print("-" * 80)
            print(f"Accuracy: {accuracy}")

        return outputs, accuracy

    def evaluate_teacher(self, val_batch_size=16, verbose=True):
        """
        Function used for evaluating student

        :param max_seq_length (int): Maximum sequence length paramter for generating dataloaders
        :param val_batch_size (int): Batch size paramter for generating dataloaders
        :param verbose (bool): True if the accuracy needs to be printed else False
        """

        self.teacher_val_loader = self._get_teacher_dataloaders(
            val_batch_size, mode="validate"
        )

        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        correct = 0
        length_of_dataset = len(self.teacher_val_loader.dataset)

        print("Evaluating teacher...")
        outputs = []

        for batch in self.teacher_val_loader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                (loss, logits) = self.teacher_model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                # out = F.softmax(logits, dim=1)
                preds = np.argmax(logits, axis=1).flatten()
                labels = label_ids.flatten()

                correct += np.sum(preds == labels)
                outputs.append(preds)

        accuracy = correct / length_of_dataset
        if verbose:
            print("-" * 80)
            print(f"Accuracy: {accuracy}")

        return outputs, accuracy
