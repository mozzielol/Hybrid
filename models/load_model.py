from models.cnn import CNN
from models.resnet import ResNet


def model_loader(multi_loss, base_model, out_dim):
    if base_model == 'cnn':
        return CNN(multi_loss, base_model, out_dim)
    elif base_model.startswith('res'):
        return ResNet(multi_loss, base_model, out_dim)