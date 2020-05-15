import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar(num_classes=10, dataset_dir='./data/cifar/', batch_size=128,
              crop=False, num_workers=2, download=True):
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

    if num_classes == 10:
        dataset = torchvision.datasets.CIFAR10
    elif num_classes == 100:
        dataset = torchvision.datasets.CIFAR100

    trainset = dataset(root=dataset_dir, train=True,
                       download=download,
                       transform=train_transform)
    testset = dataset(root=dataset_dir, train=False,
                      download=download,
                      transform=test_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True, shuffle=False)
    return trainloader, testloader


def get_mnist(num_classes=10, dataset_dir='./data/mnist/', batch_size=128,
              crop=False, num_workers=2, download=True):
    input_shape = (224, 224)
    transform = transforms.Compose([
                                torchvision.transforms.Resize(input_shape),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])

    dataset = torchvision.datasets.MNIST

    trainset = dataset(root=dataset_dir, train=True,
                       download=download,
                       transform=transform)
    testset = dataset(root=dataset_dir, train=False,
                      download=download,
                      transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True, shuffle=False)
    return trainloader, testloader


if __name__ == "__main__":
    print("MNIST")
    print(get_mnist())
    print("CIFAR100")
    print(get_cifar(100))
    print("CIFAR10")
    print(get_cifar(10))
