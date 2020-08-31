import torch
from KD_Lib.Quantization.common import Quantizer
from copy import deepcopy


class QAT_Quantizer(Quantizer):
    """
    Implementation of Quantization-Aware Training (QAT) for PyTorch models.

    :param model: (Quantizable) Model that needs to be quantized
    :type model: torch.nn.Module
    :param train_loader: DataLoader used for training
    :type train_loader: torch.utils.data.DataLoader
    :param test_loader: DataLoader used for testing
    :type test_loader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for training
    :type optimizer: torch.optim.*
    :param qconfig: Configuration used for quantization
    :type qconfig: Qconfig
    :param criterion: Loss function used for training
    :type criterion: Loss_fn
    :param device: Device used for training ("cpu" or "cuda")
    :type device: torch.device
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        qconfig=torch.quantization.get_default_qat_qconfig("fbgemm"),
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    ):
        super(QAT_Quantizer, self).__init__(
            model, qconfig, train_loader, test_loader, optimizer, criterion, device
        )

    def quantize(
        self,
        num_train_epochs=10,
        num_train_batches=10,
        param_freeze_epoch=3,
        bn_freeze_epoch=2,
    ):
        """
        Function used for quantization

        :param num_train_epochs: Number of epochs used for training
        :type num_train_epochs: int
        :param num_train_batches: Number of batches used for training
        :type num_train_batches: int
        :param param_freeze_epoch: Epoch after which quantizer parameters need to be freezed
        :type param_freeze_epoch: int
        :param bn_freeze_epoch: Epoch after which batch norm mean and variance stats are freezed
        :type bn_freeze_epoch: int
        """

        qat_model = deepcopy(self.model)
        qat_model.fuse_model()

        # optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-4)
        optimizer = deepcopy(self.optimizer)
        optimizer.params = qat_model.parameters()

        qat_model.qconfig = self.qconfig

        torch.quantization.prepare_qat(qat_model, inplace=True)

        print("Training model...")

        for epoch in range(num_train_epochs):

            print(f"Epoch {epoch}")
            loss, acc = self._train_model(qat_model, optimizer, num_train_batches)
            print(f"Training Loss: {loss} | Training Acc: {acc}")

            if epoch > param_freeze_epoch:
                qat_model.apply(torch.quantization.disable_observer)

            if epoch > bn_freeze_epoch:
                qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            self.quantized_model = torch.quantization.convert(
                qat_model.eval(), inplace=False
            )
            acc = self._evaluate_model(self.quantized_model)

            print(f"Evaluation accuracy: {acc}")

        return self.quantized_model

    def _train_model(self, model, optimizer, num_batches):
        """
        Function used for training the model

        :param model (torch.nn.Module): Model that needs to be trained
        :param optimizer (torch.optim.*): Optimizer for training
        :param num_batches (int): Number of batches used for calibration
        """

        model.to(self.device)
        model.train()

        correct = 0
        epoch_loss = 0
        cnt = 0
        len_dataset = min(
            num_batches * self.train_loader.batch_size, len(self.train_loader.dataset)
        )

        for image, target in self.train_loader:
            image, target = image.to(self.device), target.to(self.device)
            output = model(image)

            if isinstance(output, tuple):
                output = output[0]

            loss = self.criterion(output, target)

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
