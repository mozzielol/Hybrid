from models.cnn import CNN
from models.resnet import ResNet


def model_loader(dataset, multi_loss, base_model, out_dim):
    if base_model == 'cnn':
        return CNN(dataset, multi_loss, base_model, out_dim, )
    elif base_model.startswith('res'):
        return ResNet(dataset, multi_loss, base_model, out_dim)