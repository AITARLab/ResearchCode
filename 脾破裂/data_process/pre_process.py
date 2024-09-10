# -*- coding=utf-8 -*-
# @TIME 2024/3/24 20:44
# @Author: lyl
# @File: pre_process.py
# @Software:PyCharm
import numpy as np
import cv2, os, json, shutil

def rename(folder, start_id=0):
    lists = os.listdir(folder)
    map_dict = {}   # {new order: old name}存储新旧文件名的映射关系
    i = start_id
    if "阳性" in folder:
        category = "P"
    elif "阴性" in folder:
        category = "N"
    for old_name in lists:
        new_name = category+str(i)
        i += 1
        map_dict[new_name] = old_name #新旧文件名的映射关系
        old_folder_path = os.path.join(folder, old_name)
        new_folder_path = os.path.join(folder, new_name)
        os.rename(old_folder_path, new_folder_path)
    return map_dict

def clean_data(folder):
    new_json_root = "data/json"
    new_image_root = "data/images"#新的json和image文件放到"data/json"和"data/images"文件夹里
    types = ['JPG', 'jpg', 'PNG', 'png']

    patient_lists = os.listdir(folder)
    for patient in patient_lists:
        old_root = os.path.join(folder, patient)
        i = 0
        for old_name in os.listdir(old_root):
            if old_name[-4:] != "json":
                continue
            new_name = patient + "_" + str(i) + old_name[-5:]
            old_json_path = os.path.join(old_root, old_name)
            new_json_path = os.path.join(new_json_root, new_name)
            shutil.copy(old_json_path, new_json_path)

            for type in types:#用json文件找对应的文件
                old_img_path = os.path.join(old_root, old_name[:-4] + type)
                if os.path.exists(old_img_path):
                    new_img_path = os.path.join(new_image_root, new_name[:-4] + type)
                    break
            shutil.copy(old_img_path, new_img_path)
            i += 1


def json_to_mask(img_path, json_path, task):
    """

    :param img_path:
    :param json_path:
    :return:
    """
    img = cv2.imread(img_path)#读取图像文件
    mask_size = [img.shape[0], img.shape[1]]#分别是图像的高度和宽度

    with open(json_path, encoding='UTF-8') as f:
        content = json.load(f)

    shapes = content['shapes']
    o_points = []
    label_points = []
    for shape in shapes:
        category = shape['label']
        points = shape['points']
        if task == "spleen":
            if 'rupture' in category:
                continue
            elif category == "other":
                o_points.append(points)
            else:
                label_points.append(points)
        elif task == "rupture":
            if 'rupture' in category:
                label_points.append(points)
            elif category == "other":
                o_points.append(points)
            else:
                continue

    mask = np.zeros(mask_size, np.uint8)
    # 写入不规矩mask
    for i in range(len(label_points)):
        points_array = np.array(label_points[i], dtype=np.int32)
        mask = cv2.fillPoly(mask, [points_array], 255)
    # 删除多余部分
    if o_points != []:
        for i in range(len(o_points)):
            points_array = np.array(o_points[i], dtype=np.int32)
            mask = cv2.fillPoly(mask, [points_array], 0)
    return mask


def getmask(image_root, json_root, mask_root, task):
    """

    :param image_root:
    :param json_root:
    :param mask_root:
    :param task: liver or liverrupture
    :return:
    """
    for img_name in os.listdir(image_root):
        if img_name[0] == 'N' and task == "rupture":
            continue
        img_path = os.path.join(image_root, img_name)
        json_path = os.path.join(json_root, img_name[:-3] + "json")
        mask_path = os.path.join(mask_root, img_name)
        if os.path.exists(mask_path):
            continue
        mask = json_to_mask(img_path, json_path, task)
        cv2.imwrite(mask_path, mask)









