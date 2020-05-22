import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
from copy import deepcopy

class BaseClass:
    def __init__(self, teacher_model, student_model, train_loader, val_loader, optimizer_teacher, optimizer_student, loss='MSE', temp=20.0, distil_weight=0.5, device='cpu'):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.loss = loss
        self.temp = temp
        self.distl_weight = distil_weight
        self.device = device

    def train_teacher(self, epochs=20, plot_losses=True):
        '''
        Function that will be training the teacher 

        :param epochs (int): Number of epochs you want to train the teacher 
        :param plot_losses (bool): True if you want to plot the losses
        '''
        self.teacher_model.train()
        loss_arr = []

        print('Training Teacher... ')

        for ep in range(epochs):
            epoch_loss = 0.0
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.teacher_model(data)

                if isinstance(out, tuple):
                    out = out[0]

                loss = F.cross_entropy(out, label)

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()
                
                epoch_loss += loss

            loss_arr.append(epoch_loss)
            print(f'Epoch: {ep+1}, Loss: {epoch_loss}')

        if plot_losses:
            plt.plot(loss_arr)

    def train_student(self, epochs=10, plot_losses=True):
        '''
        Function that will be training the student 

        :param epochs (int): Number of epochs you want to train the teacher 
        :param plot_losses (bool): True if you want to plot the losses
        '''
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []

        print('\nTraining student...')

        for ep in range(epochs):
            epoch_loss = 0.0

            for (data, label) in self.train_loader:

                data = data.to(self.device) 
                label = label.to(self.device)

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)
                
                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss

            loss_arr.append(epoch_loss)
            print(f'Epoch {ep+1}, Loss = {epoch_loss}')

        if plot_losses:
            plt.plot(loss_arr)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        '''
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network 
        :param y_true (Tensor): True labels
        '''
        raise NotImplementedError

    def evaluate(self, teacher=True):
        '''
        Evaluate method for printing accuracies of the trained network
        
        :param teacher (bool): True if you want accuracy of the teacher network
        '''
        if teacher:
            model = deepcopy(self.teacher_model)
        else:
            model = deepcopy(self.student_model)
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0 

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                
                if isinstance(output, tuple):
                    output = output[0]
                    
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print("-"*80)
        print(f'Accuracy: {correct/length_of_dataset}')