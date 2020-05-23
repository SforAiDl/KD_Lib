import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from KD_Lib.RKD import RKDLoss
from KD_Lib.common import BaseClass
from copy import deepcopy

class TAKD(BaseClass):
    def __init__(self, teacher_model, assistant_models, student_model, assistant_train_order, train_loader, val_loader, optimizer_teacher, optimizer_assistants, optimizer_student, loss='MSE', temp=20.0, distil_weight=0.4, rkd_angle=None, rkd_dist=None, device='cpu'):
        super(TAKD, self).__init__(
            teacher_model,
            student_model,
            train_loader,
            val_loader,
            optimizer_teacher,
            optimizer_student,
            loss, 
            temp,
            distil_weight,
            device
        )
        self.assistant_models = assistant_models
        self.optimizer_assistants = optimizer_assistants
        self.assistant_train_order = assistant_train_order

        if self.loss.upper() == 'MSE':
            self.loss_fn = nn.MSELoss()

        elif self.loss.upper() == 'KL':
            self.loss_fn = nn.KLDivLoss()

        elif self.loss.upper() == 'RKD':
            self.loss_fn = RKDLoss(dist_ratio=rkd_dist, angle_ratio=rkd_angle)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        loss = (1 - self.distil_weight) * nn.CrossEntropyLoss(y_pred_student, y_true)
        loss += self.distil_weight * nn.loss_fn(
            F.log_softmax(y_pred_student / self.temp, dim=1),
            F.log_softmax(y_pred_teacher / self.temp, dim=1)
        )

        return loss

    def train_distil_model(self, teachers, model, optimizer, epochs=20, plot_losses=True, save_model=True, save_model_pth="./models/teacher.pt"):
        model.train()
        train_loss = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_model_weights = deepcopy(model.state_dict())

        if type(teachers) is not list:
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
            if epoch_acc > best_acc: 
                best_acc = epoch_acc 
                self.best_model_weights = deepcopy(model.state_dict())

            print(f'Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {epoch_acc}')

        model.load_state_dict(self.best_model_weights)
        if save_model:
            torch.save_model(model.state_dict(), save_model_pth)

    def train_assistants(self, epochs=20, plot_losses=True, save_model=True, save_dir='./models/'):
        count = 0
        for assistant in self.assistant_models:
            trainers = []
            train_order = self.assistant_train_order[count]
            for elem in train_order:
                if elem == -1:
                    trainers.append(self.teacher_model)

                else:
                    trainers.append(self.assistant_model[elem])

            self.train_distil_model(trainers, assistant, self.optimizer_assistants[count], epochs, plot_losses, save_model, save_dir+'assistant_'+count+'.pt')
            count+=1 

    def train_student(self, epochs=20, save_model=True, save_model_pth='./models/student.pth'):
        self.train_distil_model(self.assistant_models, self.student_model, self.optimizer_student, epochs, plot_losses, save_model, save_model_pth)
        