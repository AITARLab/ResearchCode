import torch.nn as nn
import torchvision.models as models


class resnet(nn.Module):
    def __init__(self, model_type=101, classes=2, pre=True):
        super(resnet, self).__init__()
        if model_type == 101:
            self.model = models.resnet101(pretrained=pre)
        elif model_type == 18:
            self.model = models.resnet18(pretrained=pre)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.linear_layer1 = nn.Linear(in_features=1000, out_features=128, bias=True)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(in_features=128, out_features=classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.linear_layer1(x)
        x = self.relu(x)
        x = self.linear_layer2(x)
        return self.softmax(x)
