import torch
from KD_Lib.Quantization.common import Quantizer
from copy import deepcopy


class QAT_Quantizer(Quantizer):
    """
    Implementation of Quantization-Aware Training (QAT) for PyTorch models.

    :param model (torch.nn.Module): (Quantizable) Model that needs to be quantized
    """

    def __init__(self, model):
        super(QAT_Quantizer, self).__init__(model)

    def quantize(
        self,
        train_loader,
        test_loader,
        model_optimizer,
        criterion,
        num_train_epochs=10,
        num_train_batches=10,
        param_freeze_epoch=3,
        bn_freeze_epoch=2,
        qconfig=torch.quantization.get_default_qat_qconfig("fbgemm"),
        device=torch.device("cpu"),
    ):
        """
        Function used for quantization

        :param train_loader(torch.utils.data.DataLoader): DataLoader used for training
        :param test_loader(torch.utils.data.DataLoader): DataLoader used for testing
        :param model_optimizer (torch.optim.*): Optimizer for training
        :param criterion(torch Loss_fn): Loss function used for calibration
        :param num_train_epochs(int): Number of epochs for training
        :param num_train_batches(int): Number of batches used for training
        :param param_freeze_epoch(int): Epoch after which quantizer parameters need to be freezed
        :param bn_free_epoch(int): Epoch after which batch norm mean and variance stats are freezed
        :param qconfig: Configuration used for quantization
        :param device(torch.device): Device used for training ("cpu" or "cuda")
        """

        qat_model = deepcopy(self.model)
        qat_model.fuse_model()

        # optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-4)
        optimizer = deepcopy(model_optimizer)
        optimizer.params = qat_model.parameters()

        qat_model.qconfig = qconfig

        torch.quantization.prepare_qat(qat_model, inplace=True)

        print("Training model...")

        for epoch in range(num_train_epochs):

            print(f"Epoch {epoch}")
            loss, acc = self._train_model(
                qat_model, train_loader, optimizer, num_train_batches, criterion, device
            )
            print(f"Training Loss: {loss} | Training Acc: {acc}")

            if epoch > param_freeze_epoch:
                self.quantized_model.apply(torch.quantization.disable_observer)

            if epoch > bn_freeze_epoch:
                self.quantized_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            self.quantized_model = torch.quantization.convert(
                qat_model.eval(), inplace=False
            )
            acc = self._evaluate_model(self.quantized_model, test_loader)

            print(f"Evaluation accuracy: {acc}")

        return self.quantized_model

    def _train_model(
        self, model, data_loader, optimizer, num_batches, criterion, device
    ):
        """
        Function used for training the model

        :param model(torch.nn.Module): Model that needs to be trained
        :param data_loader(torch.utils.data.DataLoader): DataLoader used for calibration
        :param optimizer (torch.optim.*): Optimizer for training
        :param num_batches(int): Number of batches used for calibration
        :param criterion(torch Loss_fn): Loss function used for calibration
        :param device(torch.device): Device used for training ("cpu" or "cuda")
        """

        model.to(device)
        model.train()

        correct = 0
        epoch_loss = 0
        cnt = 0
        len_dataset = min(
            num_batches * data_loader.batch_size, len(data_loader.dataset)
        )

        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = self.quantized_model(image)

            if isinstance(output, tuple):
                output = output[0]

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cnt += 1
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            epoch_loss += loss.item()

            if cnt >= num_batches:
                return epoch_loss, (correct / len_dataset)

        return epoch_loss, (correct / len_dataset)
