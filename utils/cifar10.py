# load cifar10

import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar10(batch_size=128, num_workers=2, use_augmentation=False):

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            base_transform,
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=train_transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=base_transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=base_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader
