import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 调用cuda，在这里默认的是只有一个


class resnet(nn.Module):
    def __init__(self, model_type=101, classes=2, pre=True):
        super(resnet, self).__init__()
        if model_type == 101:
            self.model = models.resnet101(pretrained=pre)
        elif model_type == 50:
            self.model = models.resnet50(pretrained=pre)
        elif model_type == 34:
            self.model = models.resnet34(pretrained=pre)
        elif model_type == 18:
            self.model = models.resnet18(pretrained=pre)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.linear_layer1 = nn.Linear(in_features=1000, out_features=128, bias=True)
        # self.relu = nn.ReLU()
        # self.linear_layer2 = nn.Linear(in_features=128, out_features=classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        # x = self.relu(self.linear_layer1(x))
        # x = self.linear_layer2(x)
        # return self.softmax(x)
        return x, self.softmax(x)


class densenet(nn.Module):
    def __init__(self, classes=2):
        super(densenet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.linear_layer1 = nn.Linear(in_features=1000, out_features=classes, bias=True)
        # self.relu = nn.ReLU()
        # self.linear_layer2 = nn.Linear(in_features=128, out_features=classes, bias=True)
        self.softmax = nn.Softmax(dim=1)
        # self.model.classifier = nn.Linear(in_features=1024, out_features=classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        # x = self.relu(self.linear_layer1(x))
        # x = self.linear_layer2(x)
        # return self.softmax(x)
        return x, self.softmax(x)


class multi_model(nn.Module):
    def __init__(self, class_num):
        super(multi_model, self).__init__()
        self.DWI_model = resnet(classes=64, model_type=18)
        self.TW1_model = resnet(classes=64, model_type=18)
        self.TW2_model = resnet(classes=64, model_type=18)
        self.relu = nn.ReLU()
        # 15 2703 2703
        # self.fc1 = nn.Linear(in_features=2703, out_features=1024, bias=True)
        # self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=True)
        # self.fc_m1 = nn.Linear(in_features=512 + 15, out_features=128, bias=True)
        self.cls_1 = nn.Linear(in_features=1000 * 3 + 2703 + 15, out_features=1024, bias=True)
        self.cls_2 = nn.Linear(in_features=1024 + 15, out_features=256, bias=True)
        self.cls_out = nn.Linear(in_features=256 + 15, out_features=class_num, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2, DWI, TW1, TW2):
        f1 = f1.float()
        f2 = f2.float()
        f1 = torch.nn.functional.normalize(f1)
        f2 = torch.nn.functional.normalize(f2)
        # f2 = self.relu(self.fc1(f2))
        # f2 = self.relu(self.fc2(f2))
        # f = torch.cat((f1, f2), 1)
        # f = self.relu(self.fc_m1(f))
        feature_DWI = None
        for dwi in DWI:
            dwi = dwi.float()
            if feature_DWI is None:
                _, feature_DWI = self.DWI_model(dwi)
            else:
                _, now = self.DWI_model(dwi)
                feature_DWI = feature_DWI + now
        leng = float(len(DWI))
        leng = torch.tensor(leng, requires_grad=True)
        leng = leng.to(device)
        if feature_DWI is not None:
            feature_DWI = feature_DWI / leng
        feature_TW1 = None
        for tw1 in TW1:
            tw1 = tw1.float()
            if feature_TW1 is None:
                _, feature_TW1 = self.TW1_model(tw1)
            else:
                _, now = self.TW1_model(tw1)
                feature_TW1 = feature_TW1 + now
        leng = float(len(TW1))
        leng = torch.tensor(leng, requires_grad=True)
        leng = leng.to(device)
        if feature_TW1 is not None:
            feature_TW1 = feature_TW1 / leng
        feature_TW2 = None
        for tw2 in TW2:
            tw2 = tw2.float()
            if feature_TW2 is None:
                _, feature_TW2 = self.TW2_model(tw2)
            else:
                _, now = self.TW2_model(tw2)
                feature_TW2 = feature_TW2 + now
        leng = float(len(TW2))
        leng = torch.tensor(leng, requires_grad=True)
        leng = leng.to(device)
        if feature_TW2 is not None:
            feature_TW2 = feature_TW2 / leng
        if feature_DWI is None:
            feature_DWI = torch.zeros(1000, requires_grad=True)
            feature_DWI = feature_DWI.unsqueeze(0)
            feature_DWI = feature_DWI.to(device)
        if feature_TW1 is None:
            feature_TW1 = torch.zeros(1000, requires_grad=True)
            feature_TW1 = feature_TW1.unsqueeze(0)
            feature_TW1 = feature_TW1.to(device)
        if feature_TW2 is None:
            feature_TW2 = torch.zeros(1000, requires_grad=True)
            feature_TW2 = feature_TW2.unsqueeze(0)
            feature_TW2 = feature_TW2.to(device)
        # print(feature_DWI.shape, feature_TW1.shape, feature_TW2.shape, f.shape)
        feature = torch.cat((feature_DWI, feature_TW1, feature_TW2, f1, f2), 1)
        feature = self.relu(self.cls_1(feature))
        feature = torch.cat((feature, f1), 1)
        feature = self.relu(self.cls_2(feature))
        feature = torch.cat((feature, f1), 1)
        feature = self.cls_out(feature)
        return  feature, self.softmax(feature) 
