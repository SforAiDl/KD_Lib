import csv, torchvision, numpy as np, random, os
from PIL import Image

from torch.utils.data import (
    Sampler,
    Dataset,
    DataLoader,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
    Subset,
)
from torchvision import transforms, datasets
from collections import defaultdict


class PairBatchSampler(Sampler):
    """
    Sampling inputs in the dataloader pairwise as mentioned in the paper "Regularizing Class-wise Predictions via Self-knowledge Distillation"
     https://arxiv.org/pdf/2003.13964.pdf

    :param dataset (torchvision.dataset): dataset
    :param batch_size (int): batch_size
    :param num_iterations (int): num_iterations
    """

    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k * self.batch_size
                batch_indices = indices[offset : offset + self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)), self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    """
    Default dataset wrapper

    :param dataset (torchvision.datasets): dataset
    :param indices (int): indices
    """

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith("0.2"):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys()) + 1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


def load_dataset(name, root, sample="default", **kwargs):
    """
    Default dataset wrapper

    :param name (string): Name of the dataset (Out of cifar10/100, imagenet, tinyimagenet, CUB200, STANFORD120, MIT67).
    :param root (string): Path to download the dataset.
    :param sample (string): Default (random) sampling as the classic pytorch dataloader or Pairwise sampling as mentioned in
    the paper "Regularizing Class-wise Predictions via Self-knowledge Distillation"
    """

    # Dataset
    if name in ["imagenet", "tinyimagenet", "CUB200", "STANFORD120", "MIT67"]:
        if name == "tinyimagenet":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(
                datasets.ImageFolder(
                    root=train_val_dataset_dir, transform=transform_train
                )
            )
            valset = DatasetWrapper(
                datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)
            )

        elif name == "imagenet":
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(
                datasets.ImageFolder(
                    root=train_val_dataset_dir, transform=transform_train
                )
            )
            valset = DatasetWrapper(
                datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)
            )

        else:
            transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            train_val_dataset_dir = os.path.join(root, name, "train")
            test_dataset_dir = os.path.join(root, name, "test")

            trainset = DatasetWrapper(
                datasets.ImageFolder(
                    root=train_val_dataset_dir, transform=transform_train
                )
            )
            valset = DatasetWrapper(
                datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)
            )

    elif name.startswith("cifar"):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if name == "cifar10":
            CIFAR = datasets.CIFAR10
        else:
            CIFAR = datasets.CIFAR100

        trainset = DatasetWrapper(
            CIFAR(root, train=True, download=True, transform=transform_train)
        )
        valset = DatasetWrapper(
            CIFAR(root, train=False, download=True, transform=transform_test)
        )
    else:
        raise Exception("Unknown dataset: {}".format(name))

    # Sampler
    if sample == "default":
        get_train_sampler = lambda d: BatchSampler(
            RandomSampler(d), kwargs["batch_size"], False
        )
        get_test_sampler = lambda d: BatchSampler(
            SequentialSampler(d), kwargs["batch_size"], False
        )

    elif sample == "pair":
        get_train_sampler = lambda d: PairBatchSampler(d, kwargs["batch_size"])
        get_test_sampler = lambda d: BatchSampler(
            SequentialSampler(d), kwargs["batch_size"], False
        )

    else:
        raise Exception("Unknown sampling: {}".format(sampling))

    train_loader = DataLoader(
        trainset, batch_sampler=get_train_sampler(trainset), num_workers=4
    )
    val_loader = DataLoader(
        valset, batch_sampler=get_test_sampler(valset), num_workers=4
    )

    return train_loader, val_loader
