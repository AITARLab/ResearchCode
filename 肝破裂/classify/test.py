# -*- coding=utf-8 -*-
# @TIME 2024/1/10 20:51
# @Author: lyl
# @File: test.py
# @Software:PyCharm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_process.dataset import *
from log.log import logger_config
import torch
from metric_function.classification_metric import *
from metric_function.visualize import *
from module.Unet import *
import time


def classification_test(log_file, model_pth, data_root, writeLog, pic_title=None, device=None):
    """

    :param log_file: 划分数据的存储文件
    :param model_pth: 模型保存的文件
    :param pic_title: 曲线图名称
    :param writeLog: log记录路径
    :return:
    """
    test_img, test_label = get_data(log_file=log_file, data_root=data_root)
    # print(len(test_img))
    test_data = image_dataset(data_image=test_img, data_label=test_label)
    test_dataloader = DataLoader(test_data, batch_size=1)

    # 加载模型测试
    model = torch.load(model_pth)
    model = model.to(device)
    true_labels = []
    predicted_labels = []
    labels_scores = []
    id = 0
    for data in test_dataloader:
        imgs, labels = data
        imgs = imgs.to(device)
        predicts = model(imgs)
        # 保存标签与分数
        np_predicts = predicts.cpu().detach().numpy()
        print(test_img[id].split("/")[-1], test_label[id], np_predicts[0][0], np_predicts[0][1])
        id += 1
        labels = list(map(int, list(labels)))
        true_labels.extend(labels)
        predicted_labels.extend(np.argmax(np_predicts, axis=1))
        for i in range(len(np_predicts)):
            labels_scores.append(np_predicts[i][1])
    # cnt_1 = 0
    # cnt_0 = 0
    # for i in range(len(true_labels)):
    #     if true_labels[i] == 1:
    #         cnt_1 += 1
    #     else:
    #         cnt_0 += 1
    metric = BinaryClassficationMetrics(true_labels, predicted_labels, labels_scores)
    ACC = metric.calculate_accuracy()
    AUC, lower, upper = metric.calculate_AUC_CI()
    F1_score = metric.calculate_f1_score()
    precision = metric.calculate_precision()
    sensitivity = metric.calculate_sensitivity()
    specificity = metric.calculate_specificity()

    logger = logger_config(log_path=writeLog,
                           logging_name="{} {}".format(model_pth, log_file))
    logger.info(
        "test: {:.4f}_AUC {:.4f}_ACC [{:.4f}:{:.4f}]_CI {:.4f}_F1_score {:.4f}_precision {:.4f}_sensitivity {:.4f}_specificity".format(
            AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity))

    if pic_title != None:
        draw_roc(true_labels, labels_scores, pic_title)
        draw_cm(true_labels, predicted_labels, pic_title, labels_name=[0, 1])
        logger.info("roc and cm image save path: {}".format(pic_title))

    return [AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity]

