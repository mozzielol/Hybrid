import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):

    def __init__(self, dataset, multi_loss, base_model, out_dim):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if multi_loss:
            # Multi-label head
            self.l1 = nn.Linear(num_ftrs, num_ftrs)
            self.l2 = nn.Linear(num_ftrs, out_dim)

        # Single-label head
        self.l3 = nn.Linear(num_ftrs, num_ftrs)
        self.l4 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")


    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, start_dim=1)

        # single-label
        x = self.l3(h)
        x = F.relu(x)
        x = self.l4(x)
        # x = F.softmax(x, dim=-1)
        return x

    def forward_multi(self, x):
        h = self.features(x)
        h = torch.flatten(h, start_dim=1)

        # multi-label
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        return x
