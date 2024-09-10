import os, json, cv2, shutil, random
import numpy as np
from sklearn.model_selection import KFold

def rename(source_root, target_root):
    """
    文件重命名与整理

    :param source_root: 数据原路径，子文件夹为[0,1,2...]
    :param target_root: 目标路径
    :return:
    """
    #目标文件路径包含image和json文件夹
    new_image_path = os.path.join(target_root, 'images')
    new_json_path = os.path.join(target_root, 'json')
    if os.path.exists(target_root) == False:
        os.mkdir(target_root)
        os.mkdir(new_image_path)
        os.mkdir(new_json_path)

    # 将图片和json文件重命名并转移至目标文件夹
    videos = os.listdir(source_root)
    for video in videos:
        video_root = os.path.join(source_root, video)
        i=0
        print("----------------------")
        print(video)
        for root, dirs, files in os.walk(video_root):
            if dirs == []:
                for f in files:
                    if f[-3:] == 'png':
                        img_name = "V"+video+'_'+str(i)+'.png'
                        img_source = os.path.join(root, f)
                        print(img_source)
                        img_target = os.path.join(new_image_path, img_name)
                        print(img_target)
                        shutil.copy(img_source, img_target)

                        json_file = f[:-4] + '.json'
                        source_json_path = os.path.join(root, json_file)

                        if os.path.exists(source_json_path):
                            json_name = "V"+video+'_'+str(i)+'.json'
                            json_source = os.path.join(source_json_path)
                            target_json_path = os.path.join(new_json_path, json_name)
                            print(json_source)
                            print(target_json_path)
                            shutil.copy(json_source, target_json_path)
                        i += 1

        print("----------------------")


def json_to_mask(json_path, mask_path):
    with open(json_path, encoding="UTF-8") as f:
        content = json.load(f)

    mask_width = content["imageWidth"]
    mask_height = content["imageHeight"]

    shapes = content["shapes"]
    o_points = []
    label_points = []
    for shape in shapes:
            category = shape['label']
            points = shape['points']
            if "rectangle" in category:
                continue
            elif category == "0":
                o_points.append(points)
            else:
                label_points.append(points)

    mask = np.zeros([mask_height, mask_width], np.uint8)
    #写入不规矩mask
    for i in range(len(label_points)):
        points_array = np.array(label_points[i], dtype=np.int32)
        mask = cv2.fillPoly(mask, [points_array], 255)
    #删除多余部分
    if o_points != []:
        for i in range(len(o_points)):
            points_array = np.array(o_points[i], dtype=np.int32)
            mask = cv2.fillPoly(mask, [points_array], 0)
    cv2.imwrite(mask_path, mask)
    print(mask_path)


def json_to_txt(json_path, txt_path):

    label = 0
    with open(json_path, encoding='UTF-8') as f:
            content = json.load(f)

    shapes = content['shapes']
    width = content["imageWidth"]
    height = content["imageHeight"]
    for shape in shapes:
        category = shape['label']
        if "rectangle" in category:
            points = shape['points']

            x_left = min(points[0][0], points[1][0])
            x_right = max(points[0][0], points[1][0])
            y_left = min(points[0][1], points[1][1])
            y_right = max(points[0][1], points[1][1])
            x_center = (x_left+x_right)/width/2
            y_center = (y_left+y_right)/height/2
            nor_width = (x_right-x_left)/width
            nor_height = (y_right-y_left)/height
            content = str(label)+' '+str(x_center)+' '+str(y_center)+' '+str(nor_width)+' '+str(nor_height)+'\n'
            with open(txt_path, 'a') as f:
                f.write(content)
        else:
            continue


def get_label(source_root):

    file_lists = os.listdir(source_root)
    for i, json_file in enumerate(file_lists):
        print(f"loading {json} {i}/{len(file_lists)}......")
        json_path = os.path.join(source_root, json_file)
        mask_path = "data/mask/" + json_file[:-4] + "png"
        label_path = "data/labels/" + json_file[:-4] + "txt"
        json_to_mask(json_path=json_path, mask_path=mask_path)
        json_to_txt(json_path=json_path, txt_path=label_path)


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


def split_data(data_root:str, total_number:int, test_percent:float=0.2, inter_test:list =[], exter_test:list=[], fold:int=0):
    """
    total_number = train + inter_test
    """
    set_lists = list(range(0, total_number))
    if test_percent == None and inter_test != []:
        pass
    elif inter_test == [] and test_percent != None:
        inter_test = random.sample(set_lists, int(total_number * test_percent))
    else:
        print("can't split the data")
    train_list = list(set(set_lists).difference(set(inter_test)))

    img_list = os.listdir(data_root)
    inter_set = []
    train_data = []
    exter_set = []
    for i, img in enumerate(img_list):
        order = int(img.split('_')[0][1:])
        # 保存内部测试集
        if order in inter_test:
            inter_set.append(img)
        # 保存训练集
        elif order in train_list:
            train_data.append(img)
        else:
            exter_set.append(img)
    
    intest_log = "log/inter_set.txt"
    save_txt(inter_set, intest_log)
    kf = KFold(n_splits=fold)  # 交叉验证
    n = 0
    for train_index, val_index in kf.split(train_data):
        train_fold, val_fold = [train_data[i] for i in train_index], [train_data[i] for i in val_index]
        train_log = f"log/{n}_fold_train.txt"
        val_log = f"log/{n}_fold_val.txt"
        n += 1
        save_txt(train_fold, train_log)
        save_txt(val_fold, val_log)
    
    if exter_set == [] or exter_test == []:
        print("there is no external test data")
    else:
        for idx, test in enumerate(exter_test):
            log_file = f"log/exter_set_{idx}.txt"
            temp = []
            for img in exter_set:
                order = int(img.split('_')[0][1:])
                if order in test:
                    temp.append(img)
            # print(temp)
            save_txt( temp,log_file)
    
    

