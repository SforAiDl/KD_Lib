import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy


def train_lstm(
    model,
    optimizer,
    train_loader,
    loss_fn=torch.nn.CrossEntropyLoss(),
    epochs=10,
    device=torch.device("cpu"),
    batch_print_freq=40,
):

    model.to(device)
    model.train()

    # training_stats = []
    loss_arr = []

    length_of_dataset = len(train_loader.dataset)

    best_acc = 0.0

    best_model_weights = deepcopy(model.state_dict())

    for ep in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(ep + 1, epochs))

        epoch_loss = 0.0
        correct = 0

        for step, batch in enumerate(train_loader):
            if step % (batch_print_freq) == 0 and not step == 0:
                print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_loader)))

            data, data_len, label = batch
            data, data_len, label = (
                data.to(device),
                data_len.to(device),
                label.to(device),
            )

            model.zero_grad()

            out = model(data, data_len).squeeze(1)

            loss = loss_fn(out, label)
            epoch_loss += loss.item()

            out = out.detach().cpu().numpy()
            label_ids = label.to("cpu").numpy()
            preds = np.argmax(out, axis=1).flatten()
            labels = label_ids.flatten()
            correct += np.sum(preds == labels)

            loss.backward()

            # #For preventing exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        epoch_acc = correct / length_of_dataset
        print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = deepcopy(model.state_dict())

        loss_arr.append(epoch_loss)

    return best_model_weights, loss_arr


def distill_to_lstm(
    model,
    optimizer,
    train_loader,
    y_pred_teacher,
    kd_loss_fn,
    epochs=10,
    device=torch.device("cpu"),
):

    model.to(device)
    model.train()

    # training_stats = []
    loss_arr = []

    length_of_dataset = len(train_loader.dataset)

    best_acc = 0.0

    best_model_weights = deepcopy(model.state_dict())

    for ep in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(ep + 1, epochs))

        epoch_loss = 0.0
        correct = 0

        for (data, data_len, label), teacher_prob in zip(train_loader, y_pred_teacher):
            data = data.to(device)
            data_len = data_len.to(device)
            label = label.to(device)

            teacher_prob = torch.tensor(teacher_prob, dtype=torch.float)
            teacher_out = teacher_prob.to(device)

            model.zero_grad()

            student_out = model(data, data_len).squeeze(1)

            loss = kd_loss_fn(student_out, teacher_out, label)

            pred = student_out.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            loss.backward()

            ##For preventing exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss

        epoch_acc = correct / length_of_dataset
        print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = deepcopy(model.state_dict())

        loss_arr.append(epoch_loss)

    return best_model_weights, loss_arr


def evaluate_lstm(model, val_loader, device=torch.device("cpu"), verbose=True):

    model.to(device)
    model.eval()

    length_of_dataset = len(val_loader.dataset)
    correct = 0
    outputs = []

    with torch.no_grad():
        for data, data_len, target in val_loader:
            data = data.to(device)
            data_len = data_len.to(device)
            target = target.to(device)
            output = model(data, data_len).squeeze(1)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            outputs.append(pred)

    if verbose:
        print("-" * 80)
        print(f"Accuracy: {correct/length_of_dataset}")

    return outputs
