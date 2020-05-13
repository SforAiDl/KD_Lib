import torch


def validate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in data_loader:
            target = target.to(device)
            data = data.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        acc = 100.0 * correct / total
        return acc


def save_model(model, optimizer, epoch, file_name):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, file_name)
