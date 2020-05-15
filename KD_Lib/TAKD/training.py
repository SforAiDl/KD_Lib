import torch.nn as nn
import torch.nn.functional as F

from KD_Lib.TAKD.utils import save_model, validate


def train_model(model, optimizer, loss_function, train_data_loader,
                val_data_loader, device, filename='best_model.pth.tar',
                epochs=1, save=True):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    iteration = 0
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for _, (data, target) in enumerate(train_data_loader):
            iteration += 1
            data = data.to(device)
            target = target.to(device)
            scores = model(data)
            loss = loss_function(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        print("epoch {}/{}".format(epoch, epochs))
        epoch_val_acc = validate(model, val_data_loader, device)
        val_acc.append(epoch_val_acc)
        print("Validation Accuracy: ", epoch_val_acc)

        if (epoch_val_acc > best_acc):
            best_acc = epoch_val_acc
            if save:
                save_model(
                    model,
                    optimizer,
                    epoch,
                    filename)

    return {
        'train_acc': train_acc,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_loss': val_loss
    }


def train_distill_model(teachers, model, optimizer, loss_function, temperature,
                        train_data_loader, val_data_loader, device, lambd,
                        filename='best_model.pth.tar', epochs=1, save=True):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    iteration = 0
    best_acc = 0.0

    if type(teachers) is not list:
        teachers = [teachers]

    for epoch in range(epochs):
        model.train()
        for _, (data, target) in enumerate(train_data_loader):
            iteration += 1
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            scores = model(data)

            teacher_output = teachers[0](data)
            for i in range(1, len(teachers)):
                teacher_output += teachers[i](data)
            teacher_output /= len(teachers)

            loss_sl = loss_function(scores, target)
            loss_kd = nn.KLDivLoss()(
                F.log_softmax(scores / temperature, dim=1),
                F.log_softmax(teacher_output / temperature, dim=1))
            loss = (1 - lambd) * loss_sl
            loss += lambd * temperature * temperature * loss_kd

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        print("epoch {}/{}".format(epoch, epochs))
        epoch_val_acc = validate(model, val_data_loader, device)
        val_acc.append(epoch_val_acc)
        print("Validation Accuracy: ", epoch_val_acc)

        if (epoch_val_acc > best_acc):
            best_acc = epoch_val_acc
            if save:
                save_model(model, optimizer, epoch, filename)

    return {
        'train_acc': train_acc,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_loss': val_loss
    }
