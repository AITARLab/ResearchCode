# -*- coding=utf-8 -*-
# @TIME 2024/3/6 20:24
# @Author: lyl
# @File: main.py.py
# @Software:PyCharm
import json
import os
import argparse
from data_process.pre_process import *
from data_process.dataset import *
from train import seg_train
from inference import *
import warnings
warnings.filterwarnings("ignore")

"""-----------------------------------预处理，无需再运行--------------------------------"""
positive = r"C:\桌面\AI\focus\脏器损伤\脾破裂\脾破裂数据\脾破裂预试验\脾破裂预试验(阳性124)"
negative = r"C:\桌面\AI\focus\脏器损伤\脾破裂\脾破裂数据\脾破裂预试验\脾破裂预试验(阴性160)"
image_root = "data/images"
json_root = "data/json"
spleen_mask_root = "data/spleen_mask"  # 脾脏的mask文件存储位置
rupture_mask_root = "data/spleenrupture_mask"  # 破裂的

#对文件改名，并且保存映射关系
# map_dict = {}
# pmap_dict = rename(positive)
# map_dict.update(pmap_dict)
# nmap_dict = rename(negative)
# map_dict.update(nmap_dict)
# with open("data/map.json", "w", encoding="utf-8") as f:
#      json.dump(map_dict, f, ensure_ascii=False, indent=4)
#
# clean_data(positive)
# clean_data(negative)
# #
# getmask(image_root, json_root, mask_root=spleen_mask_root, task="spleen")
# getmask(image_root, json_root, mask_root=rupture_mask_root,task="rupture")

# 数据划分
Plist = os.listdir(positive)
# print(len(Plist))
Nlist = os.listdir(negative)
# split_data(image_root, Plist, Nlist, test_percent=0.2, fold=5)
split_data_rupture(rupture_mask_root, Plist, test_percent=0.2, fold=5)

# 模型分割脾脏区域
# pth = "weights/fcn_resnet101_0fold_23epoch_0.9472ACC_0.9257IOU_0.9600DICE.pth"
# data_root = "data/images"
# for img_id in os.listdir("data/spleenrupture_mask"):
#     print(img_id)
#     img_path = os.path.join(data_root, img_id)
#     seg_inference(pth, img_path)

"""--------------------------------------------------------------------------------------"""

"""---------------------------------------train------------------------------------------"""
# fold = 5
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 分割肝脏
# seg_train(model_name="UNet", fold=fold, device=device, task="spleen_seg")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=50, task="spleen_seg")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=101, task="spleen_seg")
# seg_train(model_name="COD", fold=fold, device=device, model_type=101, task="spleen_seg")
# # # 分割损伤
# seg_train(model_name="UNet", fold=fold, device=device, task="one_step_seg")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=50, task="one_step_seg")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=10, task="one_step_seg")
# seg_train(model_name="COD", fold=fold, device=device, model_type=10, task="one_step_seg")
# # 分割损伤
# seg_train(model_name="UNet", fold=fold, device=device, task="two_step_seg")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=50, task="two_step_seg")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=10, task="two_step_seg")
# seg_train(model_name="COD", fold=fold, device=device, model_type=10, task="two_step_seg")
