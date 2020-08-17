import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformers import BertForSequenceClassification, AdamW, BertTokenizer

from KD_Lib.common import BaseClass
from KD_Lib.KD.text.utils.bert import train_bert, evaluate_bert, get_bert_dataloader
from KD_Lib.KD.text.utils.lstm import train_lstm, evaluate_lstm, distill_to_lstm

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Bert2LSTM(BaseClass):
    """
    Implementation of Knowledge distillation from the paper "Distilling Task-Specific
    Knowledge from BERT into Simple Neural Networks" https://arxiv.org/pdf/1903.12136.pdf

    :param student_model (torch.nn.Module): Student model
    :param distill_train_loader (torch.utils.data.DataLoader): Student Training Dataloader for distillation
    :param distill_val_loader (torch.utils.data.DataLoader): Student Testing/validation Dataloader
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param train_df (pandas.DataFrame): Dataframe for training the teacher model
    :param val_df (pandas.DataFrame): Dataframe for validating the teacher model
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher. If None, AdamW is used.
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
        train_df,
        val_df,
        num_classes=2,
        seed=42,
        loss_fn=nn.MSELoss(),
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        self.set_seed(42)

        self.train_df, self.val_df = train_df, val_df
        
        teacher_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = num_classes,
            output_attentions = False,
            output_hidden_states = False,
        )

        optimizer_teacher = AdamW(teacher_model.parameters(),
                                  lr = 2e-5,
                                  eps = 1e-8
                                  )
        
        
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


        super(Bert2LSTM, self).__init__(
            teacher_model,
            student_model,
            distill_train_loader,
            distill_val_loader,
            optimizer_teacher,
            None,
            loss_fn,
            temp,
            distil_weight,
            device,
            log,
            logdir,
        )

        self.optimizer_student = torch.optim.Adam(self.student_model.parameters(), lr=2e-4)


    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _get_teacher_dataloaders(self, max_seq_length=128, batch_size=16, mode="train"):
        """
        Helper function for generating dataloaders for the teacher
        """
        df = self.val_df if (mode == "validate") else self.train_df

        return get_bert_dataloader(df, self.bert_tokenizer, max_seq_length, batch_size, mode)
        
    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation
        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model 
        :param y_true (torch.FloatTensor): Original label
        """

        # soft_teacher_out = F.softmax(y_pred_teacher / self.temp, dim=0)
        # soft_student_out = F.log_softmax(y_pred_student / self.temp, dim=0)

        soft_teacher_out = y_pred_teacher
        soft_student_out = y_pred_student
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        loss = (1 - self.distil_weight) * self.criterion_ce(soft_student_out, y_true)
        loss += self.distil_weight * self.criterion_mse(soft_student_out, soft_teacher_out)
        return loss

    def train_teacher(
        self,
        epochs=1,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/teacher.pt",
        max_seq_length=128,
        train_batch_size=16,
        batch_print_freq=40
    ):
        """
        Function that will be training the teacher 
        :param epochs (int): Number of epochs you want to train the teacher 
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        :param max_seq_length (int): Maximum sequence length paramter for generating dataloaders
        :param train_batch_size (int): Batch size paramter for generating dataloaders
        """
        self.teacher_train_loader = self._get_teacher_dataloaders(max_seq_length, train_batch_size, mode="train")

        best_weights, loss_arr = train_bert(self.teacher_model, 
                                           self.optimizer_teacher, 
                                           self.teacher_train_loader, 
                                           epochs, 
                                           self.device, 
                                           batch_print_freq)

        self.teacher_model.load_state_dict(best_weights)
        
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

        self.teacher_distill_loader = self._get_teacher_dataloaders(batch_size=self.train_loader.batch_size,mode="distill")

        y_pred_teacher = evaluate_bert(self.teacher_model, self.teacher_distill_loader, self.device)

        best_weights, loss_arr = distill_to_lstm(self.student_model, 
                                                 self.optimizer_student, 
                                                 self.train_loader, 
                                                 y_pred_teacher,
                                                 self.calculate_kd_loss,
                                                 epochs, 
                                                 self.device)

        
        self.student_model.load_state_dict(best_weights)
        
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        
        if plot_losses:
            plt.plot(loss_arr)

    def evaluate_student(self, verbose=True):
        return evaluate_lstm(self.student_model, self.val_loader, self.device)

    def evaluate_teacher(self, max_seq_length=128, val_batch_size=16, verbose=True):

        self.teacher_val_loader = self._get_teacher_dataloaders(max_seq_length, val_batch_size, mode="validate")
        return evaluate_bert(self.teacher_model, self.teacher_val_loader, self.device)
        