# -*- coding = utf-8 -*-
# @time:2024/7/16 19:10
# Author:lyh
# @File:test1.py
# @Software:PyCharm
import shutil

import matplotlib.pyplot as plt
from ultralytics import YOLO
import os


def merge_datasets(train_folds, val_fold, temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'labels/val'), exist_ok=True)

    # 合并训练数据
    for fold in train_folds:
        for subset in ['images', 'labels']:
            src_dir = os.path.join(fold, subset)
            dst_dir = os.path.join(temp_dir, os.path.join(subset,'train'))
            for file_name in os.listdir(src_dir):
                shutil.copy(os.path.join(src_dir, file_name), dst_dir)

    # 合并验证数据
    for subset in ['images', 'labels']:
        src_dir = os.path.join(val_fold, subset)
        dst_dir = os.path.join(temp_dir, os.path.join(subset,'val'))
        for file_name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, file_name), dst_dir)
merge_datasets(['dataset/fold0','dataset/fold1','dataset/fold2','dataset/fold3'],'dataset/fold4','temp_dir')

