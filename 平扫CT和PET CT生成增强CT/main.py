# -*- coding=utf-8 -*-
# @TIME 2024/3/27 20:10
# @Author: lyl
# @File: main.py.py
# @Software:PyCharm
import os

import matplotlib.pyplot as plt
import numpy as np
from log.draw_analysis import draw_mask
from data_process.preprocess import *
from data_process.dataset import *
from test import *
from PIL import Image
from train import *
from inference import *

"""-------------------------------------预处理-----------------------------------------"""
# source_data_root = r"C:\桌面\AI\focus\CT与PET CT肿瘤血管分割\原始数据"
# 病人名称与序号对应
# map_dict = rename(source_data_root)
# with open("data/map.json", "w", encoding="utf-8") as f:
    #  json.dump(map_dict, f, ensure_ascii=False, indent=4)

# 图像整理与转移
# clean_data(r"C:\桌面\AI\focus\CT与PET CT肿瘤血管分割\原始数据\12")

# image_root = "data/image"
# json_root = "data/json"
# mask_root = "data/mask"
# mask_map = {"M": 1}
# task = '_'.join(mask_map.keys())
# mask_root = "data/{}_mask".format(task)
# getmask(image_root, json_root, mask_root,  mask_map)
# task = ["liver", "cancer"]  # liver:分割肝脏，cancer：分割病变区域并分类
# patient = os.listdir(source_data_root)
# split_data(data_root=mask_root, PatientList=patient, test_percent=0.2, task=task, fold=5, seed=54)
"""--------------------------------训练--------------------------------"""
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
fold = 5
task = ["liver","J","M","cancerhgyynm "]
for t in task:
    seg_train(model_name="UNet", fold=fold, task=t, device=device)
    seg_train(model_name="fcn_resnet", fold=fold, task=t, device=device, model_type=50)
    seg_train(model_name="fcn_resnet", fold=fold, task=t, device=device, model_type=101)
