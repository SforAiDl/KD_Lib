import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy


def train_bert(
    model,
    optimizer,
    train_loader,
    epochs=10,
    device=torch.device("cpu"),
    batch_print_freq=40,
):
    """
    Function that will be training BERT 
    :param epochs (int): Number of epochs you want to train the teacher 
    :param plot_losses (bool): True if you want to plot the losses
    :param save_model (bool): True if you want to save the teacher model
    :param save_model_pth (str): Path where you want to store the teacher model
    :param max_seq_length (int): Maximum sequence length paramter for generating dataloaders
    :param train_batch_size (int): Batch size paramter for generating dataloaders
    """
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

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            loss, logits = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            epoch_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            preds = np.argmax(logits, axis=1).flatten()
            labels = label_ids.flatten()
            correct += np.sum(preds == labels)

            loss.backward()

            # For preventing exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        epoch_acc = correct / length_of_dataset
        print(f"Loss: {epoch_loss} | Accuracy: {epoch_acc}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = deepcopy(model.state_dict())

        loss_arr.append(epoch_loss)

    return best_model_weights, loss_arr


def distill_to_bert():
    raise NotImplementedError


def evaluate_bert(model, val_loader, device=torch.device("cpu"), verbose=True):
    model.to(device)
    model.eval()

    correct = 0
    length_of_dataset = len(val_loader.dataset)

    outputs = []

    with torch.no_grad():
        for batch in val_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            (loss, logits) = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            preds = np.argmax(logits, axis=1).flatten()
            labels = label_ids.flatten()
            correct += np.sum(preds == labels)

            outputs.append(preds)

    if verbose:
        print("-" * 80)
        print(f"Accuracy: {correct/length_of_dataset}")

    return outputs
