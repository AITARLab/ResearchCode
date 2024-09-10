import random

import pandas as pd
import numpy as np
import random as rd

now_df = open("./PDL1_train.txt", 'r')

id_list = []
for i in now_df:
    i = i.strip()
    id_list.append(i)

random.shuffle(id_list)

for fold in range(4):
    train_txt = open("./PDL1_train_fold{}.txt".format(fold), "w")
    val_txt = open("./PDL1_val_fold{}.txt".format(fold), "w")
    for i in range(76):
        if fold * 19 <= i < (fold + 1) * 19:
            val_txt.write("{}\n".format(id_list[i]))
        else:
            train_txt.write("{}\n".format(id_list[i]))
    train_txt.close()
    val_txt.close()


# df = pd.read_excel('./Clinical_Features/Clinical_Data_Processed.xlsx')
#
# pdl1_0 = []
# pdl1_1 = []
#
# vegf_0 = []
# vegf_1 = []
#
# for index, row in df.iterrows():
#     id = row['patients']
#     pdl1 = int(row['pdl1'])
#     vegf = int(row['vegf'])
#     if pdl1 == 1:
#         pdl1_1.append(id)
#     else:
#         pdl1_0.append(id)
#     if vegf == 1:
#         vegf_1.append(id)
#     else:
#         vegf_0.append(id)
#
# rd.seed = 42
# rd.shuffle(pdl1_0)
# rd.shuffle(pdl1_1)
# rd.shuffle(vegf_0)
# rd.shuffle(vegf_1)
#
# print(len(pdl1_0), len(pdl1_1))
# print(len(vegf_0), len(vegf_1))
#
# split_data_pdl1_test = open("PDL1_test.txt", 'w', newline='')
# split_data_pdl1_train = open("PDL1_train.txt", 'w', newline='')
# split_data_vegf_test = open("VEGF_test.txt", 'w', newline='')
# split_data_vegf_train = open("VEGF_train.txt", 'w', newline='')
#
# for i in range(10):
#     split_data_pdl1_test.write('{}\n'.format(pdl1_1[i]))
#     split_data_pdl1_test.write('{}\n'.format(pdl1_0[i]))
#     split_data_vegf_test.write('{}\n'.format(vegf_1[i]))
#     split_data_vegf_test.write('{}\n'.format(vegf_0[i]))
#
# for i in range(10, len(pdl1_0)):
#     split_data_pdl1_train.write('{}\n'.format(pdl1_0[i]))
# for i in range(10, len(pdl1_1)):
#     split_data_pdl1_train.write('{}\n'.format(pdl1_1[i]))
# for i in range(10, len(vegf_0)):
#     split_data_vegf_train.write('{}\n'.format(vegf_0[i]))
# for i in range(10, len(vegf_1)):
#     split_data_vegf_train.write('{}\n'.format(vegf_1[i]))
#
# split_data_pdl1_train.close()
# split_data_vegf_train.close()
# split_data_vegf_test.close()
# split_data_pdl1_test.close()
