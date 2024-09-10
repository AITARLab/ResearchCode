import os
import shutil
import random


def copy_file(src_file, dest_file):
    shutil.copy(src_file, dest_file)


def solve():
    img_list = os.listdir('./balanced_test/images')
    id_list = [i for i in range(len(img_list))]
    random.shuffle(id_list)
    cnt = 0
    id_index = {}
    for img_file in img_list:
        name = img_file.split('.')[0]
        id_index[name] = id_list[cnt]
        copy_file('./balanced_test/images/{}'.format(img_file), './doctor_test/images/{}.jpg'.format(id_list[cnt]))
        copy_file('./balanced_test/labels/{}.txt'.format(name), './doctor_test/answer/{}.txt'.format(id_list[cnt]))
        cnt += 1
    model_list = os.listdir('./balanced_test/result')
    for img_file in model_list:
        name = img_file.split('.')[0]
        index = id_index[name]
        copy_file('./balanced_test/result/{}'.format(img_file), './doctor_test/model_output/{}.jpg'.format(index))
    print('done!')


def resort():
    txt_list = os.listdir("./doctor_test/answer")
    txt_ans = {}
    for txt_file in txt_list:
        t = open("doctor_test/answer/{}".format(txt_file), "r")
        line = t.readline()
        txt_ans[int(txt_file.split('.')[0])] = line.split(' ')[0]
        # print("{}:{}".format(txt_file, line.split(' ')[0]))
    for i in range(406):
        print("{}:{}".format(i, txt_ans[i]))


if __name__ == "__main__":
    # solve()
    resort()
