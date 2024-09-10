# -*- coding=utf-8 -*-
# @TIME 2024/3/25 11:29
# @Author: lyl
# @File: preprocess.py
# @Software:PyCharm
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, json, shutil
from PIL import Image


def clean_data(folder_root):
    new_json_root = "data/json"
    new_image_root = "data/image"
    for root, patients, files in os.walk(folder_root):
        if patients != []:
            continue
        for file in files:
            if file[-4:] != 'json':
                continue
            new_name = root.split("\\")[-1] + "_" + file
            old_json_path = os.path.join(root, file)
            new_json_path = os.path.join(new_json_root, new_name)
            shutil.copy(old_json_path, new_json_path)
            old_img_path = os.path.join(root, file[:-4] + "jpg")
            new_img_path = os.path.join(new_image_root, new_name[:-4] + "jpg")
            shutil.copy(old_img_path, new_img_path)


def json_to_mask(img_path, json_path, mask_count):
    """

    :param img_path:
    :param json_path:
    :return:
    """
    img = cv2.imread(img_path)
    mask_size = [img.shape[0], img.shape[1]]
    mask_map = {"股内侧肌": 1, "缝匠肌": 2, "半膜肌": 3, "半腱肌": 4, "股二头肌": 5, "股薄肌": 6, "骨薄肌": 6}
    mask_point = {"股内侧肌": [], "缝匠肌": [], "半膜肌": [], "半腱肌": [], "股二头肌": [], "股薄肌": [], "骨薄肌": []}
    # mask_count = {"股内侧肌": 0, "缝匠肌": 0, "半膜肌": 0, "半腱肌": 0, "股二头肌": 0, "股薄肌": 0, "骨薄肌": 0}

    with open(json_path, encoding='UTF-8') as f:
        content = json.load(f)

    shapes = content['shapes']
    for shape in shapes:
        category = shape['label']
        points = shape['points']
        mask_point[category].append(points)
        mask_count[category] += 1

    # 写入不规则mask
    mask = np.zeros(mask_size, np.uint8)
    for key in mask_point.keys():
        label_points = mask_point[key]
        for i in range(len(label_points)):
            points_array = np.array(label_points[i], dtype=np.int32)
            mask = cv2.fillPoly(mask, [points_array], mask_map[key])
    return mask, mask_count


def check_mask(mask_path):
    """

    :param mask_path:
    :return:
    """
    mask = cv2.imread(mask_path)
    mask[mask != 0] = 255
    plt.imshow(mask)
    plt.show()


def getmask(image_root, json_root, mask_root):
    """

    :param image_root:
    :param json_root:
    :param mask_root:
    :return:
    """
    mask_count = {"股内侧肌": 0, "缝匠肌": 0, "半膜肌": 0, "半腱肌": 0, "股二头肌": 0, "股薄肌": 0, "骨薄肌": 0}
    for img_name in os.listdir(image_root):
        img_path = os.path.join(image_root, img_name)
        json_path = os.path.join(json_root, img_name[:-3] + "json")
        mask_path = os.path.join(mask_root, img_name[:-3] + 'png')
        mask, mask_count = json_to_mask(img_path, json_path, mask_count)
        mask = Image.fromarray(mask, 'L')
        mask.save(mask_path)
    # print(mask_count)
    return mask