import json, os
import pandas as pd
from sklearn.model_selection import KFold

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

test_patient = ['曹锐', '杨福均', '陈铃轩', '李乔良', '肖宜桥', '魏松阳', '米绍林', '朱福彪', '安明益', '何书文', '王浩然', '刘明联', '李厚珍', '唐天平', '王椿', '兰华书', '张英']
task = "liverrupture_surgery"
file_path = "/home/user/user/2/ML/data/mydata - 副本 (3).xlsx"
test_dict = {}

df = pd.read_excel(file_path)
lable_dict = dict(zip(df['patient'], df['surgery']))

with open("/home/user/user/2/injury/data/map.json","r",encoding="utf-8") as f:
    name_dict = json.load(f)

train_dict = {}
test_dict = {}

for pid in name_dict.keys():
    pname = name_dict[pid]
    if pname not in lable_dict.keys():
        continue
    if pname in test_patient:
        test_dict[pid] = lable_dict[pname]
    else:
        train_dict[pid] = lable_dict[pname]

image_root = "/home/user/user/2/injury/data/images"
train_data = []
test_data = []
for file in os.listdir(image_root):
    pid = file.split("_")[0]
    if pid in train_dict.keys():
        str = "{} {}".format(file, train_dict[pid])
        train_data.append(str)
    elif pid in test_dict.keys():
        str = "{} {}".format(file, test_dict[pid])
        test_data.append(str)

save_txt(test_data, f"../log/{task}_test.txt")
kf = KFold(n_splits=5)  # 交叉验证
fold = 0
for train_index, val_index in kf.split(train_data):
    train_fold, val_fold = [train_data[i] for i in train_index], [train_data[i] for i in val_index]
    train_log = f"../log/{task}_{fold} fold_train.txt"
    val_log = f"../log/{task}_{fold} fold_val.txt"
    fold += 1
    save_txt(train_fold, train_log)
    save_txt(val_fold, val_log)

