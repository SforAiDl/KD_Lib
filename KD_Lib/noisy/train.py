import torch
import torch.nn.functional as F
from .utils import add_noise
import random


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