# -*- coding=utf-8 -*-
# @TIME 2024/1/17 17:29
# @Author: lyl
# @File: train.py.py
# @Software:PyCharm
# from torch.utils.data import DataLoader
import numpy as np
from module.CNN import *
from module.Densenet import *
from module.Resnet import *
from module.Unet import *
from module.fcn_resnet import *
from log.log import logger_config
from test import *
from data_process.dataset import *
import torch.optim
import torch.nn as nn
from metric_function.classification_metric import *
from metric_function.visualize import *


def classification_train(task, fold, model_name, model_type=None, device=None):
    """
    :param fold:
    :param model_name:
    :param model_type:
    :param device:
    :return:
    """
    writeLog = "log/classfication_log.txt"
    logger = logger_config(log_path=writeLog,
                           logging_name=f"{task} {model_name} classification train")

    if model_name == "CNN":
        model = CNN(n_classes=2)
    elif model_name == "Resnet":
        model = resnet(model_type=model_type)
        model_name = "{}_{}".format(model_name, model_type)
    elif model_name == "Densenet":
        model = densenet()
    else:
        return
    # 设置参数
    learning_rate = 5e-5
    img_root = "/home/user/user/2/injury/data/images"
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    epoch = 120
    batchsize = 8
    # 每个元素为[AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity]
    val_result = []
    test_result = []
    model = model.to(device)

    for fn in range(fold):
        train_str = "log/{}_{} fold_train.txt".format(task, fn)
        val_str = "log/{}_{} fold_val.txt".format(task, fn)
        train_img, train_label = get_data(log_file=train_str,data_root=img_root)
        val_img, val_label = get_data(log_file=val_str, data_root=img_root)
        train_data = image_dataset(data_image=train_img, data_label=train_label)
        val_data = image_dataset(data_image=val_img, data_label=val_label)
        train_dataloader = DataLoader(train_data, batch_size=batchsize)
        val_dataloader = DataLoader(val_data, batch_size=batchsize)
        # logger.info("load data success!")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        best_train_value = 0
        best_test_value = 0
        train_step = 0

        for n in range(epoch):
            logger.info("{} {} fold {} epoch train {}".format("-" * 20, fn, n, "-" * 20))
            model.train()
            for data in train_dataloader:
                imgs, labels = data
                imgs = imgs.to(device)
                predicts = model(imgs)
                # print(predicts)
                optimizer.zero_grad()
                labels = list(map(int, list(labels)))
                loss = loss_fn(predicts.cpu(), torch.tensor(labels, dtype=torch.long))
                loss.backward()
                optimizer.step()

                if train_step % 100 == 0:
                    logger.info("after {} steps training, the loss is {:.4f}".format(train_step, loss))
                train_step += 1

            # 验证集
            true_labels = []
            predicted_labels = []
            labels_scores = []
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    imgs, labels = data
                    imgs = imgs.to(device)
                    predicts = model(imgs)
                    # 保存标签与分数
                    np_predicts = predicts.cpu().detach().numpy()
                    labels = list(map(int, list(labels)))
                    true_labels.extend(labels)
                    predicted_labels.extend(np.argmax(np_predicts, axis=1))
                    for i in range(len(np_predicts)):
                        labels_scores.append(np_predicts[i][1])

                # 评价指标计算
                metric = BinaryClassficationMetrics(true_labels, predicted_labels, labels_scores)
                ACC = metric.calculate_accuracy()
                AUC, lower, upper = metric.calculate_AUC_CI()
                F1_score = metric.calculate_f1_score()
                precision = metric.calculate_precision()
                sensitivity = metric.calculate_sensitivity()
                specificity = metric.calculate_specificity()
                # val_result.append([AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity])
                logger.info(
                    "val: {:.4f}_AUC {:.4f}_ACC [{:.4f}:{:.4f}]_CI {:.4f}_F1_score {:.4f}_precision {:.4f}_sensitivity {:.4f}_specificity".format(
                        AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity))

            # 保存当前最好的值
            temp_value = 0.8 * AUC + 0.2 * ACC
            if temp_value > best_train_value:
                best_train_value = temp_value
                train_best = [AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity]

                if os.path.exists("weights") == False:
                    os.mkdir("weights")
                save_path = "weights/{}_{}_{}fold_{}epoch_{:.4f}AUC_{:.4f}ACC.pth".format(task, model_name, fn, n, AUC,
                                                                                          ACC)
                torch.save(model, save_path)
                logger.info("weight save path:{}".format(save_path))
                # val 混淆矩阵图与roc
                if os.path.exists("results") == False:
                    os.mkdir("results")
                roc_and_cm_title = "result1/{}_{}_{}fold_{}epoch".format(task, model_name, fn, n)
                draw_roc(true_labels, labels_scores, roc_and_cm_title + "_val")
                draw_cm(true_labels, predicted_labels, roc_and_cm_title + "_val", labels_name=[0, 1])
                logger.info("roc and cm image save path: {}_val".format(roc_and_cm_title))
                # 内部测试集
                log_intest_data = f"log/{task}_test.txt"
                test_temp = classification_test(log_file=log_intest_data, model_pth=save_path,
                            pic_title=roc_and_cm_title+"_test", data_root=img_root, writeLog=writeLog, device=device)
                if 0.8 * test_temp[0] + 0.2 * test_temp[1] > best_test_value:
                    test_best = test_temp

        # 每折最佳值
        logger.info("{} {}fold best performance {}".format("-" * 20, fn, "-" * 20))
        val_result.append(train_best)
        test_result.append(test_best)
        logger.info('VAL:   AUC:{:.4f}  ACC:{:.4f}  CI:[{:.4f}, {:.4f}] F1_score:{:.4f}  precision:{:.4f}   sensitivity:{:.4f}  specificity:{:.4f}'.format(*train_best))
        logger.info('TEST:   AUC:{:.4f}  ACC:{:.4f}  CI:[{:.4f}, {:.4f}] F1_score:{:.4f}  precision:{:.4f}   sensitivity:{:.4f}  specificity:{:.4f}'.format(*test_best))

    # AUC, ACC, lower, upper, F1_score, precision, sensitivity, specificity指标均值
    val_mean = np.mean(np.array(val_result), axis=0)
    test_mean = np.mean(np.array(test_result), axis=0)
    logger.info("{} avg performance {}".format("-" * 20, "-" * 20))
    logger.info('VAL:   AUC:{:.4f}  ACC:{:.4f}  CI:[{:.4f}, {:.4f}] F1_score:{:.4f}  precision:{:.4f}   sensitivity:{:.4f}  specificity:{:.4f}'.format(*val_mean))
    logger.info('TEST:   AUC:{:.4f}  ACC:{:.4f}  CI:[{:.4f}, {:.4f}] F1_score:{:.4f}  precision:{:.4f}   sensitivity:{:.4f}  specificity:{:.4f}'.format(*test_mean))

