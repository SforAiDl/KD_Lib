import torch.nn.functional as F
import matplotlib.pyplot as plt


def train_teacher(model, train_loader, optimizer, epochs, device):

    model.train()
    loss_arr = []

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

    plt.plot(loss_arr)
    plt.show()


def train_student(teacher_model, student_model, train_loader, optimizer, loss_fn, epochs=10, temp=20, distil_weight=0.7):

    teacher_model.eval()
    student_model.train()
    loss_arr = []

    for e in range(epochs):
        epoch_loss = 0

        for (data, label) in train_loader:

            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()

            soft_label = F.softmax(teacher_model(data)/temp)
            out = student_model(data)
            soft_out = F.softmax(out/temp)

            loss = (1 - distil_weight) * F.cross_entropy(out, label) + (distil_weight) * loss_fn(soft_label, soft_out)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

        loss_arr.append(epoch_loss)
        print(f'Epoch {e+1} loss = {epoch_loss}')

    plt.plot(loss_arr)
    plt.show()
