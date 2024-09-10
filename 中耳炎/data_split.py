import os,shutil
# test = os.listdir("balanced_test\images")
# full = os.listdir(r"dataset_full\full\images")
# del_list = list(set(test) & set(full))
# # print(len(del_list))
# # print(len(test))
# # print(len(full))
# image_root = r"dataset_full\full\images"
# txt_root = r"dataset_full\full\labels"
# for img_file in  del_list:
#     img_path = os.path.join(image_root, img_file)
#     txt_file = img_file.split(".")[0]+".txt"
#     txt_path = os.path.join(txt_root, txt_file)
#     os.remove(img_path)
#     os.remove(txt_path)
# def split_into_folds(data_list):
#     print("split")
#     fold_size = len(data_list) // 5
#     new_fold = []
#     for i in range(5):
#         if i == 4:
#             temp = data_list[i * fold_size:]
#         else:
#             temp = data_list[i * fold_size:(i + 1) * fold_size]
#         new_fold.append(temp)
#     return new_fold

# def move(data_list):
#     old_img_root = r"dataset_full\full\images"
#     old_txt_root = r"dataset_full\full\labels"

#     for i in range(len(data_list)):
#         new_img_root = f"dataset_full/fold{i}/images"
#         new_txt_root = f"dataset_full/fold{i}/labels"
#         if os.path.exists(new_img_root) == False:
#             os.makedirs(new_img_root)
#         if os.path.exists(new_txt_root) == False:
#             os.makedirs(new_txt_root)

#         for img_file in data_list[i]: 
#             txt_file = img_file.split(".")[0]+".txt"
#             shutil.move(os.path.join(old_img_root, img_file), os.path.join(new_img_root, img_file))
#             # 移动文本文件
#             shutil.move(os.path.join(old_txt_root, txt_file), os.path.join(new_txt_root, txt_file))
#     # return [data_list[i * fold_size:(i + 1) * fold_size] for i in range(5)]
# count = {'0': 265, '1': 862, '2': 478}
# zero_list = []
# one_list = []
# two_list = []
# print(len(full))
# for file in full:
#     fid = file.split("_")[0]
#     # print(file)
#     if fid == "0" :
#         print(file)
#         zero_list.append(file) 
#     elif fid == "1":
#         one_list.append(file)
#     elif fid == "2":
#         two_list.append(file)

# # print(len(zero_list))
# two = split_into_folds(two_list)
# move(two)
count_object = {}
count_img = {}
# root = "balanced_test\labels"
for i in range(5):
    root = r"dataset_full\fold{}\labels".format(i)
    for file in os.listdir(root):
        fid = file.split("_")[0]
        if fid not in count_img.keys():
            count_img[fid] = 0
        if fid not in count_object.keys():
            count_object[fid] = 0

        with open(os.path.join(root, file),"r") as f:
            lines = f.readlines()
        non_empty_lines = [line for line in lines if line.strip()]

        count_img[fid] += 1
        count_object[fid] += len(non_empty_lines)

        if len(non_empty_lines) != 1:
            print(os.path.join(root, file))

print(count_img)
print(count_object)