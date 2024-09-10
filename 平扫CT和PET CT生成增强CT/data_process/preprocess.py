# -*- coding=utf-8 -*-
# @TIME 2024/3/27 20:08
# @Author: lyl
# @File: preprocess.py
# @Software:PyCharm
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, json, shutil, difflib, re
from PIL import Image


def rename(folder):
    lists = os.listdir(folder)
    map_dict = {}  # {new order: old name}
    for i in range(len(lists)):
        new_name = str(i)
        old_name = lists[i]
        map_dict[new_name] = old_name
        old_folder_path = os.path.join(folder, old_name)
        new_folder_path = os.path.join(folder, new_name)
        os.rename(old_folder_path, new_folder_path)
    return map_dict


def rename_same(folder):
    path_list = os.listdir(folder)
    for p in path_list:
        split_list = p.split(".")
        type = split_list[-1]
        new_id = split_list[0][3:-1]
        new_name = new_id + "." + type
        # new_name = split_list[-1]
        old_path = os.path.join(folder, p)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(old_path)
        print("-------")
        print(new_path)


def clean_data(folder):
    """

    :param folder: 文件夹内包含两个文件[ct,融合图]
    :return:
    """
    new_json_root = "data/json"
    new_image_root = "data/image"
    # pic_types = ['png', 'JPG', 'jpg', 'PNG']
    folder_list = os.listdir(folder)
    patient_id = folder.split("\\")[-1]
    
    i = 0

    type1 = "ct" if "CT" in folder_list[0] else "fusion"
    open_path1 = os.path.join(folder, folder_list[0])
    path1_list = os.listdir(open_path1)
    type2 = "ct" if "CT" in folder_list[1] else "fusion"
    open_path2 = os.path.join(folder, folder_list[1])
    path2_list = os.listdir(open_path2)


    for file in path1_list:
        if file[-4:] != 'json':
            continue

        new_name1 = patient_id + "_" + type1 + "_" + str(i)
        new_name2 = patient_id + "_" + type2 + "_" + str(i)
        old_json_name = file
        old_image_name = file[:-4] + "png"

        old_json_path1 = os.path.join(open_path1, old_json_name)
        new_json_path1 = os.path.join(new_json_root, new_name1+".json")
        # print(old_json_path1, new_json_path1)
        shutil.copy(old_json_path1, new_json_path1)
        old_image_path1 = os.path.join(open_path1, old_image_name)
        new_image_path1 = os.path.join(new_image_root, new_name1 + '.png')
        # print(old_image_path1, new_image_path1)
        shutil.copy(old_image_path1, new_image_path1)

        # file = file[5:].replace(')',"")
        file = file.replace("102652", "103031")
        print(file)
        if file in path2_list:
            old_json_path2 = os.path.join(open_path2, file)
            new_json_path2 = os.path.join(new_json_root, new_name2 + ".json")
            # print(old_json_path2, new_json_path2)
            shutil.copy(old_json_path2, new_json_path2)
            old_image_path2 = os.path.join(open_path2, file[:-4] + "png")
            new_image_path2 = os.path.join(new_image_root, new_name2 + '.png')
            # print(old_image_path2, new_image_path2)
            shutil.copy(old_image_path2, new_image_path2)
        i += 1


def json_to_mask(img_path, json_path):
    """

    :param img_path:
    :param json_path:
    :return:
    """
    img = cv2.imread(img_path)
    mask_size = [img.shape[0], img.shape[1]]
    mask_map = {"liver": 1, "cancer": 1, "benign": 2}
    mask_point = {"liver": [], "cancer": [], "benign": []}

    with open(json_path, encoding='UTF-8') as f:
        content = json.load(f)

    shapes = content['shapes']
    for shape in shapes:
        category = shape['label']
        points = shape['points']
        if category not in mask_point.keys():
            continue
        mask_point[category].append(points)

    # 写入不规则mask
    mask = np.zeros(mask_size, np.uint8)
    for key in mask_point.keys():
        label_points = mask_point[key]
        for i in range(len(label_points)):
            points_array = np.array(label_points[i], dtype=np.int32)
            mask = cv2.fillPoly(mask, [points_array], mask_map[key])
    return mask


def getmask(image_root, json_root, mask_root):
    """

    :param image_root:
    :param json_root:
    :param mask_root:
    :return:
    """
    for img_name in os.listdir(image_root):
        img_path = os.path.join(image_root, img_name)
        json_path = os.path.join(json_root, img_name[:-3] + "json")
        mask_path = os.path.join(mask_root, img_name[:-3] + 'npy')
        mask = json_to_mask(img_path, json_path)
        # plt.imshow(mask)
        # plt.show()
        np.save(mask_path, mask)
        # mask = Image.fromarray(mask, 'L')
        # mask.save(mask_path)


