import os
import torch
import random
import torchvision
import numpy as np
import torch.utils.data
from torchvision import transforms
from data_loader.data_augment import Cutout, CIFAR10Policy, ImageNetPolicy


def DataLoader(data_type, data_set, batch_size, data_augment, num_workers):
    if data_type == 'CIFAR10':
        data_path = os.path.join(data_set, data_type)
        dataset = load_cifar10(data_path, batch_size, data_augment, num_workers)
    elif data_type == 'CIFAR100':
        data_path = os.path.join(data_set, data_type)
        dataset = load_cifar100(data_path, batch_size, data_augment, num_workers)
    else:
        raise (ValueError('Unavailable dataset'))
    return dataset


def load_cifar10(data_path: str, batch_size: int, data_augment: bool, num_workers: int):
    if data_augment:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), CIFAR10Policy(),
             transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def load_cifar100(data_path: str, batch_size: int, data_augment: bool, num_workers: int):
    # CIFAR100 normalization stats; augment mirrors CIFAR10 (RandomCrop+HFlip+AutoAugment CIFAR10Policy).
    # NOTE: verify CIFAR100 augment/normalize against arXiv:2511.13050 Appendix when available.
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    if data_augment:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), CIFAR10Policy(),
             transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transforms = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader