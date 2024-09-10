import os, shutil, json, cv2
import numpy as np


def clean_data(root, img_root, json_root):

    if os.path.exists(img_root) == False:
        os.makedirs(img_root)

    if os.path.exists(json_root) == False:
        os.makedirs(json_root)
    
    patients = os.listdir(root)
    map_dict = {}

    for id, patient in enumerate(patients):
        map_dict[str(id)] = patient
        dir_path = os.path.join(root, patient)
        i = 0

        for file in os.listdir(dir_path):
            if file[-4:] != "json":
                continue

            old_json_path = os.path.join(dir_path, file)
            new_json_path = os.path.join(json_root, f"{id}_{i}.json")
            shutil.copy2(old_json_path, new_json_path)

            old_img_path = os.path.join(dir_path, file.replace("json", "png"))
            new_img_path = os.path.join(img_root, f"{id}_{i}.png")
            shutil.copy2(old_img_path, new_img_path)

            i += 1

    with open("data/map.json", "w", encoding="utf-8") as f:
            json.dump(map_dict, f, ensure_ascii=False, indent=4)


def is_contained(inner, outer, mask_size):
    inner_mask = np.zeros(mask_size, np.uint8)
    points_array = np.array(inner, dtype=np.int32)
    inner_mask = cv2.fillPoly(inner_mask, [points_array], 255)

    outer_mask = np.zeros(mask_size, np.uint8)
    points_array = np.array(outer, dtype=np.int32)
    outer_mask = cv2.fillPoly(outer_mask, [points_array], 255)

    intersection = cv2.bitwise_and(inner_mask, outer_mask)
    return np.array_equal(intersection, inner_mask)

def json_to_mask(img_path, json_path):
    """

    :param img_path:
    :param json_path:
    :return:
    """
    img = cv2.imread(img_path)
    mask_size = [img.shape[0], img.shape[1]]

    with open(json_path, encoding='UTF-8') as f:
        content = json.load(f)

    shapes = content['shapes']
    o_points = []
    label_points = []   
    for shape in shapes:
        points = shape['points']
        label_points.append(points)
    

    if len(label_points) == 2:
        print(f"{json_path} have 2 label")

        if is_contained(label_points[0], label_points[1], mask_size):
            o_points.append(label_points[0])
            del label_points[0]
        elif is_contained(label_points[1], label_points[0], mask_size):
            o_points.append(label_points[1])
            del label_points[1]
        else:
            print("no contained")

    elif len(label_points) > 2:
        print(json_path)

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

def getmask(image_root, json_root, mask_root):
    """

    :param image_root:
    :param json_root:
    :param mask_root:
    :return:
    """
    if os.path.exists(mask_root) == False:
        os.mkdir(mask_root)

    for img_name in os.listdir(image_root):
        img_path = os.path.join(image_root, img_name)
        json_path = os.path.join(json_root, img_name[:-3] + "json")
        mask_path = os.path.join(mask_root, img_name)
        mask = json_to_mask(img_path, json_path)
        cv2.imwrite(mask_path, mask)