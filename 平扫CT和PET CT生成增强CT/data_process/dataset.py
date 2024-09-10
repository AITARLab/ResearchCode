# -*- coding=utf-8 -*-
# @TIME 2024/3/25 11:29
# @Author: lyl
# @File: dataset.py
# @Software:PyCharm
import os.path
from torchvision import transforms
from PIL import Image
import numpy as np
import torch, cv2, random
import torch.utils.data as td
from sklearn.model_selection import KFold


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


def split_data(data_root: str, PatientList: list, test_percent: int, task: str, fold: int = 0, seed: int = 38):
    test_number = int(len(PatientList) * test_percent)
    random.seed(seed)
    test = random.sample(PatientList, test_number)
    print(f"test number:{test_number}")

    test_data = []
    train_data = []
    for data in os.listdir(data_root):
        if data.split("_")[0] in test:
            test_data.append(data)
        else:
            train_data.append(data)
    print(f"test img number :{len(test_data)}, train img number :{len(train_data)}")

    test_log = f"log/{task}_inter_set.txt"
    save_txt(test_data, test_log)

    kf = KFold(n_splits=fold)  # 5折交叉验证
    fold = 0
    for train_index, val_index in kf.split(train_data):
        train_fold, val_fold = [train_data[i] for i in train_index], [train_data[i] for i in val_index]
        train_log = f"log/{task} {fold} fold train.txt"
        val_log = f"log/{task} {fold} fold val.txt"
        fold += 1
        save_txt(train_fold, train_log)
        save_txt(val_fold, val_log)


def get_data(log_file, mask_root, img_root):
    """

    :param log_file: 保存有图片名称的txt文件
    :param mask_root:
    :param tasl: seg
    :return: 返回存有图像和mask路径的两个列表
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    img_set = []
    mask_set = []
    for line in lines:
        mask_name = line.strip()
        img_set.append(os.path.join(img_root, mask_name.replace('npy','png')))
        mask_set.append(os.path.join(mask_root, mask_name))
    return img_set, mask_set


def seg_loader(img_path, mask_path):
    """

    :param img_path:
    :param mask_path:
    :param task:
    :return:
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mask = np.load(mask_path)
    mask = cv2.resize(mask, (512, 512))
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize([512, 512]),
                                transforms.Normalize(mean, std)])
    return trans(img), torch.from_numpy(mask)


class img_dataset(td.Dataset):
    def __init__(self, data_image, data_mask, task):
        self.image = data_image
        self.mask = data_mask
        self.task = task

    def __getitem__(self, index):
        img, mask = seg_loader(img_path=self.image[index], mask_path=self.mask[index])
        return img, mask

    def __len__(self):
        return len(self.image)