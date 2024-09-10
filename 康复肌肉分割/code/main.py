# -*- coding=utf-8 -*-
# @TIME 2024/3/25 11:30
# @Author: lyl
# @File: main.py.py
# @Software:PyCharm
import os
from data_process.preprocess import *
from data_process.dataset import *
from train import *

"""------------------------------------预处理-------------------------------------------------------------"""
# clean_data(folder_root)
# json_to_mask("data/image/20190129_00000002.jpg","data/json/20190129_00000002.json")
# check_mask("data/mask/20190613_00000003.jpg")
# image_root = "data/image"
# json_root = "data/json"
# mask_root = "data/mask"
# print(len(os.listdir(mask_root)))
# getmask(image_root, json_root, mask_root)
# split_data(data_root="data/image", PatientList=os.listdir(folder_root), test_percent=0.2, fold = fold)
"""------------------------------------------------------------------------------------------------------"""
folder_root = r"C:\桌面\AI\focus\康复肌肉\勾画示例"
fold = 5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=50)
seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=101)
seg_train(model_name="UNet", fold=fold, device=device)