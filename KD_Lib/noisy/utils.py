import torch
import torch.optim as optim
import torch.nn as nn

from KD_Lib.RKD import RKDLoss
from .train import train_teacher, train_student

def eval(model, data_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    total = len(data_loader.dataset)
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'{correct}/{total} correct')
    print(f'The accuracy of the model is {correct/total}')

def add_noise(x, variance = 0.1):
    return x*(1 + (variance**0.5) * torch.randn_like(x))

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