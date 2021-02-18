import torch
import torchvision
import torchvision.transforms as transforms

"""
Dataloader
    - input: None
    - output: trainloader, testloader
"""


class Dataloader:
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             ])

        trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)
        return trainloader, testloader


