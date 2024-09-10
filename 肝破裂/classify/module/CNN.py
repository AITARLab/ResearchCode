import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))

        self.fc = nn.Sequential(
            nn.Linear(32 * 128 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)

        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(-1, 32 * 128 * 128)
        x = self.fc(x)
        return self.softmax(x)
