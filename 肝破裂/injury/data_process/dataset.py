import os.path
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import torch, cv2, random
import torch.utils.data as td
from sklearn.model_selection import KFold
from log.draw_analysis import *
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


def split_data(data_root: str, Plist:list, Nlist:list, test_percent: int, fold: int = 0, seed=38):
    min_number = min(len(Plist), len(Nlist))
    test_number = int(min_number * test_percent)
    # print(f"test_number:{test_number}")
    random.seed(seed)
    test_P = random.sample(Plist, test_number)
    test_N = random.sample(Nlist, int(test_number/3)+1)
    print(f"test Positive patient number:{test_number},test Negative patient number:{int(test_number/3)}")
    test = test_N + test_P
    train_P = list(set(Plist).difference(set(test_P)))
    train_N = list(set(Nlist).difference(set(test_N)))
    print(f"train patient number of Positive:{len(train_P)}, train patient number of Negative:{len(train_N)}")
    train_data = []
    test_data = []
    test_p_count = 0
    test_n_count = 0
    train_p_count = 0
    train_n_count = 0

    for data in os.listdir(data_root):
        idx = data.split("_")[0]
        if idx in test:
            test_data.append(data)
            if idx in test_P:
                test_p_count += 1
            else:
                test_n_count += 1
        else:
            train_data.append(data)
            if idx in train_P:
                train_p_count += 1
            else:
                train_n_count += 1
    print(f"test img number :{len(test_data)}, train img number :{len(train_data)}")
    print(f"test_p img_count:{test_p_count},test_n img_count:{test_n_count},train_p img_count:{train_p_count}, train_n img_count:{train_n_count}")
    test_log = "log/inter_set.txt"
    save_txt(test_data, test_log)

    kf = KFold(n_splits=fold)  # 交叉验证
    fold = 0
    for train_index, val_index in kf.split(train_data):
        train_fold, val_fold = [train_data[i] for i in train_index], [train_data[i] for i in val_index]
        train_log = f"log/{fold} fold train.txt"
        val_log = f"log/{fold} fold val.txt"
        fold += 1
        save_txt(train_fold, train_log)
        save_txt(val_fold, val_log)


def slpit_data_liverrupture(data_root: str, Plist:list, test_percent: float, seed: int =38, fold: int = 0):
    number = len(Plist)
    test_number = int(number * test_percent)
    print(f"test_number:{test_number}")
    random.seed(seed)
    test = random.sample(Plist, test_number)
    train = list(set(Plist).difference(set(test)))
    print(f"train patient number of positive:{len(train)}")
    train_data = []
    test_data = []

    for data in os.listdir(data_root):
        idx = data.split("_")[0]
        if idx in test:
            test_data.append(data)
        else:
            train_data.append(data)

    test_log = "log/liverrupture_inter_set.txt"
    save_txt(test_data, test_log)

    kf = KFold(n_splits=fold)  # 交叉验证
    fold = 0
    for train_index, val_index in kf.split(train_data):
        train_fold, val_fold = [train_data[i] for i in train_index], [train_data[i] for i in val_index]
        train_log = f"log/liverrupture {fold} fold train.txt"
        val_log = f"log/liverrupture {fold} fold val.txt"
        fold += 1
        save_txt(train_fold, train_log)
        save_txt(val_fold, val_log)


def get_data(log_file, mask_root, img_root, degree="all"):
    """

    :param log_file: 保存有图片名称的txt文件
    :param mask_root:
    :param tasl: seg
    :param degree: 损伤程度
    :return: 返回存有图像和mask路径的两个列表
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    if degree != "all":
        df = pd.read_excel("data/副本肝破裂阳性（临床数据）汇总(1)(1).xlsx")
        filter_df = df[df['分级'].isin(degree)]
        names = filter_df['肝破裂'].tolist()
        with open('data/map.json',"r", encoding="utf-8") as f:
            map_dict = json.load(f)

    img_set = []
    mask_set = []
    for line in lines:
        img_name = line.strip()
        if degree != "all" and map_dict[img_name.split('_')[0]] not in names:
            continue
        img_set.append(os.path.join(img_root, img_name))
        mask_set.append(os.path.join(mask_root, img_name))
    return img_set, mask_set



def seg_loader(img_path, mask_path, mode=None):
    """

    :param img_path:
    :param mask_path:
    :return:
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    open_img = Image.open(img_path).convert("RGB")
    img = np.array(open_img)
    mask = np.array(Image.open(mask_path))
    mask = torch.from_numpy(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) / 255)
    if mode == "train":
        liver_mask_root = img_path.replace("images", "mask").split(".")[0]
        types = ['JPG', 'jpg', 'PNG', 'png']
        for type in types:
                liver_mask_path = liver_mask_root + "." + type
                if os.path.exists(liver_mask_path) == True:
                    break
        liver_mask = np.array(Image.open(liver_mask_path))
        mask_array = liver_mask[:, :, np.newaxis]
        img = np.array(image_bitwise_mask(open_img, mask_array))
    return trans(img), mask


def classify_loader(img_path):
    p_id = img_path.split("/")[-1].split("\\")[-1].split("_")[0]


class img_dataset(td.Dataset):
    def __init__(self, data_image, data_mask, mode=None):
        self.image = data_image
        self.mask = data_mask
        self.mode = mode

    def __getitem__(self, index):
        img, mask = seg_loader(img_path=self.image[index],
                               mask_path=self.mask[index],
                               mode=self.mode)
        return img, mask

    def __len__(self):
        return len(self.image)



