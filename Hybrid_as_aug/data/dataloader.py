import os
import torch
import torchvision
import torchvision.transforms as transforms

"""
Dataloader
    - input: None
    - output: trainloader, testloader
"""
"""
TODO:
    - turn on/off normalisation
    - imagenet augmentation
"""


class Dataloader:
    def __init__(self, data_dir, batch_size, num_workers, name, augmentation, standardization):
        self.data_dir = data_dir
        self.dataset = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.standardization = standardization

    def get_data_loaders(self):
        transform_test = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()] \
            if self.dataset.lower().startswith('image') else [transforms.ToTensor()]

        if self.augmentation:
            if self.dataset.lower() == 'cifar10':
                crop = transforms.RandomCrop(32, padding=4)
            elif self.dataset.lower() == 'stl10':
                crop = transforms.RandomCrop(96, padding=4)
            else:   # imagenet
                crop = transforms.RandomResizedCrop(224)

            transform_train = [crop, transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        else:
            transform_train = transform_test

        if self.standardization:
            standardization_transform = transforms.Normalize(
                mean=(0.485, 0.456, 0.406) if self.dataset.lower().startswith('image') else (0.4914, 0.4822, 0.4465),
                std=(0.229, 0.224, 0.225) if self.dataset.lower().startswith('image') else (0.2023, 0.1994, 0.2010)
            )
            transform_train.append(standardization_transform)
            transform_test.append(standardization_transform)

        transform_train, transform_test = transforms.Compose(transform_train), transforms.Compose(transform_test)

        try:
            loader = getattr(self, self.dataset)
            return loader(transform_train, transform_test)
        except AttributeError:
            raise ValueError('dataset is not available')

    def cifar10(self, transform_train, transform_test):
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader

    def imagenet(self, transform_train, transform_test):
        imgs = torchvision.datasets.ImageFolder(root='/home/fantasie/Pictures/ImageNet/imagenet_images')
        train_num = round(.9 * len(imgs))
        train_set, val_set = torch.utils.data.random_split(
            imgs, [train_num, len(imgs) - train_num], generator=torch.Generator().manual_seed(2021)
        )
        trainloader = torch.utils.data.DataLoader(Subset(train_set, transform_train),
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(Subset(val_set, transform_test),
                                                 batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.num_workers)
        return trainloader, testloader

    def stl10(self, transform_train, transform_test):
        trainset = torchvision.datasets.STL10(root=self.data_dir, split='train',
                                              download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.STL10(root=self.data_dir, split='test',
                                             download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader

    def imagenette(self, transform_train, transform_test):
        dataset_dir = '/home/fantasie/Pictures/ImageNet/imagenette2-320'
        trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader

    def imagewoof(self, transform_train, transform_test):
        dataset_dir = '/home/fantasie/Pictures/ImageNet/imagewoof2-320'
        trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader


class Subset(torch.utils.data.DataLoader):
    """
    Create a Dataset using a subset obtained from random_split, with specified transforms applied
    Reference: https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/2
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
