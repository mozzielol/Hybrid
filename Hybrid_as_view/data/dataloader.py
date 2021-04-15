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
    def __init__(self, data_dir, batch_size, num_workers, name, augmentation):
        self.data_dir = data_dir
        self.dataset = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation

    def get_data_loaders(self):
        if self.augmentation:
            crop = transforms.RandomCrop(32, padding=4) if self.dataset == 'cifar10' else transforms.RandomCrop(96,
                                                                                                                padding=4)
            transform_train = transforms.Compose([
                crop,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
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
        data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        imgs = torchvision.datasets.ImageFolder(root='/home/fantasie/Pictures/ImageNet/imagenet_images',
                                                transform=data_transform)
        train_num = round(.9 * len(imgs))
        train_set, val_set = torch.utils.data.random_split(imgs, [train_num, len(imgs) - train_num])
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)
        testloader = torch.utils.data.DataLoader(val_set,
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
