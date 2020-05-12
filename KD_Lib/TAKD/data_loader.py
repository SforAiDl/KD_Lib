import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10(dataset_dir='./data/cifar10', batch_size=128, crop=False,
                num_workers=2, download=True):
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.507, 0.487, 0.441],
                                            std=[0.267, 0.256, 0.276])
                                        ])
    if crop is True:
        train_transform = transforms.Compose([
                                            transforms.RandomCrop(32,
                                                                  padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.507, 0.487, 0.441],
                                                std=[0.267, 0.256, 0.276])
                                            ]
                                        )
    else:
        train_transform = test_transform

    trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
                                            download=download,
                                            transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                           download=download,
                                           transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    print("CIFAR10")
    print(get_cifar10())
