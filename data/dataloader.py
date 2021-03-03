import torch
import torchvision
import torchvision.transforms as transforms

"""
Dataloader
    - input: None
    - output: trainloader, testloader
"""


class Dataloader:
    def __init__(self, batch_size, num_workers, name):
        self.dataset = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             ])
        try:
            loader = getattr(self, self.dataset)
            return loader(transform)
        except AttributeError:
            raise ValueError('dataset is not available')

    def cifar10(self, transform):
        trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader

    def imagenet(self, transform):
        trainset = torchvision.datasets.ImageNet(root='./datasets', split='train',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.ImageNet(root='./datasets', split='test',
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader


    def stl10(self, transform):
        trainset = torchvision.datasets.STL10(root='./datasets', split='train',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.STL10(root='./datasets', split='test',
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader
