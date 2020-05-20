import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

from KD_Lib.noisy.utils import add_noise, eval
from KD_Lib.RKD import RKDLoss


def train_teacher(model, train_loader, optimizer, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    loss_arr = []

    print('Training teacher - \n')

    for e in range(epochs):
        epoch_loss = 0
        for (data, label) in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1} loss = {epoch_loss}')

def train_student(teacher_model, student_model, train_loader, optimizer, 
                  loss_fn, epochs=20, alpha = 0.7, variance=1):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model.eval()
    student_model.train()
    loss_arr = []

    print('Training student - \n')

    for e in range(epochs):
        epoch_loss = 0

        for (data, label) in train_loader:

            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()

            t_logits = teacher_model(data)
            if random.uniform(0,1) <= alpha:
                t_logits = add_noise(t_logits)

            out = student_model(data)

            loss = loss_fn(out, t_logits)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1} loss = {epoch_loss}')

def run_experiment(train_loader, test_loader, teacher_model, student_model, 
                   epochs, lr, optimizer, loss, alpha, variance,
                   rkd_dist, rkd_angle):

    if optimizer.upper() == 'SGD':
        t_optimizer = optim.SGD(teacher_model.parameters(), lr, momentum=0.9)
        s_optimizer = optim.SGD(student_model.parameters(), lr, momentum=0.9)
    elif optimizer.upper() == 'Adam':
        t_optimizer = optim.Adam(teacher_model.parameters(), lr)
        s_optimizer = optim.Adam(student_model.parameters(), lr)

    if loss.upper() == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss.upper() == 'KL':
        loss_fn = nn.KLDivLoss()
    elif loss.upper() == 'RKD':
        loss_fn = RKDLoss(dist_ratio=rkd_dist, angle_ratio=rkd_angle)

    train_teacher(teacher_model, train_loader, t_optimizer, epochs)
    eval(teacher_model, test_loader)

    train_student(teacher_model, student_model, train_loader, s_optimizer,
                  loss_fn, epochs, alpha, variance)
    eval(student_model, test_loader)