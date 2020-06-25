import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, random_split)
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForSequenceClassification, AdamW, BertTokenizer

from KD_Lib.common import BaseClass
from KD_Lib.Bert2Lstm.utils import df_to_dataset, batch_to_inputs, set_seed

import numpy as np
from copy import deepcopy

class Bert2LSTM(BaseClass):
    """
    Original implementation of Knowledge distillation from the paper "Distilling Task-Specific
    Knowledge from BERT into Simple Neural Networks" https://arxiv.org/pdf/1903.12136.pdf

    :param student_model (torch.nn.Module): Student model
    :param distill_train_loader (torch.utils.data.DataLoader): Student Training Dataloader for distillation
    :param distill_val_loader (torch.utils.data.DataLoader): Student Testing/validation Dataloader
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param train_df (pandas.DataFrame): Dataframe for training the teacher model
    :param val_df (pandas.DataFrame): Dataframe for validating the teacher model
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher. If None, AdamW is used.
    :param loss (str): Loss used for training
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """
    
    def __init__(self, student_model, distill_train_loader, distill_val_loader, optimizer_student, train_df, val_df,
                 optimizer_teacher=None,loss='MSE', temp=20.0, distil_weight=0.5, device='cpu', log=False, logdir='./Experiments'):

        set_seed(42)

        self.train_df, self.val_df = train_df, val_df
        teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        if optimizer_teacher is None:
            optimizer_teacher = AdamW(teacher_model.parameters())
        
        super(Bert2LSTM, self).__init__(
            teacher_model,
            student_model,
            distill_train_loader,
            distill_val_loader,
            optimizer_teacher,
            optimizer_student,
            loss,
            temp,
            distil_weight,
            device,
            log,
            logdir
        )
        if self.loss.upper() == 'MSE':
            self.loss_fn = nn.MSELoss()

        elif self.loss.upper() == 'KL':
            self.loss_fn = nn.KLDivLoss()

    def _get_teacher_dataloaders(self, max_seq_length=128, train_batch_size=16, val_batch_size=16, mode='train'):
        '''
        Helper function for generating dataloaders for the teacher
        '''
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if mode == 'validate':
            val_dataset = df_to_dataset(self.val_df, bert_tokenizer, max_seq_length)
            val_sampler = SequentialSampler(val_dataset)
            val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size) 
            return val_loader     

        dataset = df_to_dataset(self.train_df, bert_tokenizer, max_seq_length)

        if mode == 'distill':
            distill_sampler = SequentialSampler(dataset)
            distill_loader = DataLoader(dataset, sampler=distill_sampler, batch_size=train_batch_size)
            return distill_loader

        elif mode == 'train':
            train_sampler = RandomSampler(dataset)
            train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size)
            return train_loader
        
    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation
        :param y_pred_student (torch.FloatTensor): Prediction made by the student model
        :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model 
        :param y_true (torch.FloatTensor): Original label
        """
        
        soft_teacher_out = F.softmax(y_pred_teacher/self.temp, dim=0)
        soft_student_out = F.softmax(y_pred_student/self.temp, dim=0)

        loss = (1-self.distil_weight) * F.cross_entropy(soft_student_out,
                                                       y_true)
        loss += self.distil_weight * self.loss_fn(soft_teacher_out,
                                                 soft_student_out)
        return loss

    def train_teacher(self, epochs=1, plot_losses=True, save_model=True, save_model_pth="./models/teacher.pt",
        max_seq_length=128, train_batch_size=16):
        '''
        Function that will be training the teacher 
        :param epochs (int): Number of epochs you want to train the teacher 
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        :param max_seq_length (int): Maximum sequence length paramter for generating dataloaders
        :param train_batch_size (int): Batch size paramter for generating dataloaders
        '''
        self.teacher_train_loader = self._get_teacher_dataloaders(max_seq_length, train_batch_size, mode='train')

        self.teacher_model.to(self.device)
        self.teacher_model.train()
        loss_arr = []
        length_of_dataset = len(self.teacher_train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())
       
        print('Training Teacher... ')

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            for batch in self.teacher_train_loader:
                batch = tuple(item.to(self.device) for item in batch)
                inputs = batch_to_inputs(batch)
                outputs = self.teacher_model(**inputs)

                loss = outputs[0]
                out = torch.softmax(outputs[1],dim=1)

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(batch[3].view_as(pred)).sum().item()

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()
                
                epoch_loss += loss

            epoch_acc = correct/length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

            if self.log:
                self.writer.add_scalar('Training loss/Teacher', epoch_loss, epochs)
                self.writer.add_scalar('Training accuracy/Teacher', epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(f'Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def train_student(self, epochs=10, plot_losses=True, save_model=True, save_model_pth='./models/student.pth'):
        '''
        Function that will be training the student 
        :param epochs (int): Number of epochs you want to train the teacher 
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        '''
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        y_pred_teacher = None

        self.teacher_distill_loader = self._get_teacher_dataloaders(mode='distill')

        for batch in self.teacher_distill_loader:
            batch = tuple(t.to(self.device) for t in batch)
            inputs = batch_to_inputs(batch)

            with torch.no_grad():
                outputs = self.teacher_model(**inputs)
                logits = outputs[1]

                logits = logits.cpu().numpy()

                if y_pred_teacher is None:
                    y_pred_teacher = logits
                else:
                    y_pred_teacher = np.vstack((y_pred_teacher, logits))

        self.student_model.train()
        
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        print('\nTraining student...')

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0

            for (data, label), bert_prob in zip(self.train_loader,y_pred_teacher):

                data = data.to(self.device)
                teacher_out = bert_prob.to(device) 
                label = label.to(self.device)

                student_out = self.student_model(data.t()).squeeze(1)
                
                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                if isinstance(student_out, tuple):
                    student_out = student_out[0]

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss

            epoch_acc = correct / length_of_dataset
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self.best_student_model_weights = deepcopy(self.student_model.state_dict())

            if self.log:
                self.writer.add_scalar('Training loss/Student', epoch_loss, epochs)
                self.writer.add_scalar('Training accuracy/Student', epoch_acc, epochs)

            loss_arr.append(epoch_loss)
            print(f'Epoch: {ep+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

