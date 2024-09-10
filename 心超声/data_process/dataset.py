import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torch, cv2, random
import torch.utils.data as td
from sklearn.model_selection import KFold
# from log.draw_analysis import *
import json

def save_txt(content: list, save_path: str):
    """

    :param content:
    :param save_path:
    :return:
    """
    with open(save_path, "w", encoding='utf-8') as f:
        for c in content:
            f.write(c)
            f.write("\n")
    print(f"{save_path} saved!")

def split_data(data_root: str, fold: int = 0, seed=74):
    if os.path.exists("log") == False:
        os.makedirs("log")

    random.seed(seed)
    train_data = os.listdir(data_root)

    kf = KFold(n_splits=fold)  # 交叉验证
    fold = 0
    for train_index, val_index in kf.split(train_data):
        train_fold, val_fold = [train_data[i] for i in train_index], [train_data[i] for i in val_index]
        train_log = f"log/{fold} fold train.txt"
        val_log = f"log/{fold} fold val.txt"
        fold += 1
        save_txt(train_fold, train_log)
        save_txt(val_fold, val_log)

def get_data(log_file, mask_root, img_root):
    """

    :param log_file: 保存有图片名称的txt文件
    :param mask_root:
    :param tasl: seg
    :param degree: 损伤程度
    :return: 返回存有图像和mask路径的两个列表
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    img_set = []
    mask_set = []
    for line in lines:
        img_name = line.strip()
        img_set.append(os.path.join(img_root, img_name))
        mask_set.append(os.path.join(mask_root, img_name))
    return img_set, mask_set


def seg_loader(img_path, mask_path):
    """

    :param img_path:
    :param mask_path:
    :return:
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                transforms.Normalize((0.5), (0.5))])
    open_img = Image.open(img_path).convert("RGB")
    img = np.array(open_img)
    mask = np.array(Image.open(mask_path))
    mask = torch.from_numpy(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) / 255)
    return trans(img), mask


class img_dataset(td.Dataset):
    def __init__(self, data_image, data_mask):
        self.image = data_image
        self.mask = data_mask

    def __getitem__(self, index):
        img, mask = seg_loader(img_path=self.image[index],
                               mask_path=self.mask[index])
        return img, mask

    def __len__(self):
        return len(self.image)
