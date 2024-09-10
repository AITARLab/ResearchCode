# -*- coding=utf-8 -*-
# @TIME 2024/1/16 17:08
# @Author: lyl
# @File: dataset.py
# @Software:PyCharm
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.utils.data as td
import numpy as np
from torchvision import transforms
from PIL import Image
import time

def get_data(log_file, data_root):
    imgs = []
    labels = []
    
    with open(log_file, 'r') as f:
        for line in f.readlines():
            img_id, label = line.strip("\n").split(" ")
            img_path = os.path.join(data_root,img_id)
            imgs.append(img_path)
            labels.append(label)

    return imgs, labels


def loader(img_path):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512])])
    # open_img = Image.open(img_path).convert("RGB")
    # img = np.array(open_img)
    img = cv2.imread(img_path)
    mask = cv2.imread(img_path.replace("images","liverrupture"))
    result = cv2.bitwise_and(img, mask)
    # return trans(img)
    return trans(result)

class image_dataset(td.Dataset):
    def __init__(self, data_image, data_label):
        """

        :param data_image:
        :param data_label_or_mask:
        :param mode:
        :param n_classes:
        """
        self.label = data_label
        self.image = data_image

    def __getitem__(self, index):
        img = loader(img_path=self.image[index])
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)