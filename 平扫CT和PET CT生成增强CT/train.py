# -*- coding=utf-8 -*-
# @TIME 2023/11/22 22:16
# @Author: lyl
# @File: train.py
# @Software:PyCharm

import torch.optim
from Model import fcn_resnet, Unet
from torch.utils.data import DataLoader
from data_process.dataset import *
from metric_function.seg_metric import SegmentationMetric
import torch.nn as nn
import torchvision.models as models
from log.log import *
from test import *
import torch.nn.functional as F


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def seg_train(model_name, fold, device, task, model_type=None, continue_train = False):
    """

    :param model_name:
    :param fold:
    :param device:
    :param model_type:
    :return:
    """
    in_channels = 3
    num_classes = 2 
    # 模型选择
    if model_name == "fcn_resnet":
        model = fcn_resnet.fcn(in_channels=in_channels, num_classes=num_classes, model_type=model_type)
        model_name = "{}{}".format(model_name, model_type)
    elif model_name == "UNet":
        model = Unet.UNet(in_channels=in_channels, num_classes=num_classes)

    # 设置参数
    model = model.to(device)
    record_log = "seg_log.txt"
    inter_log = f"log/{task}_inter_set.txt"
    logger = logger_config(log_path=record_log, logging_name=f"{model_name} {task} train")
    image_root = "data/image"
    mask_root = f"data/{task}_mask"
    loss_fn = nn.CrossEntropyLoss().to(device)
    metric = SegmentationMetric(numClass=num_classes)
    train_value = []
    test_value = []
    learning_rate = 5e-4
    epoch = 80
    batchsize = 8
    checkpoint_path = "checkpoint.pth"

    for i in range(fold):
        if continue_train and os.path.exists(checkpoint_path):
                model, optimizer, i, loss = load_checkpoint_model(checkpoint_path, model, optimizer)
                continue_train = False

        train_str = "log/{} {} fold train.txt".format(task, i)
        val_str = "log/{} {} fold val.txt".format(task, i)
        train_img, train_mask = get_data(log_file=train_str, mask_root=mask_root, img_root=image_root)
        val_img, val_mask = get_data(log_file=val_str, mask_root=mask_root, img_root=image_root)
        train_data = img_dataset(data_image=train_img, data_mask=train_mask, task=task)
        val_data = img_dataset(data_image=val_img, data_mask=val_mask, task=task)
        train_dataloader = DataLoader(train_data, batch_size=batchsize)
        val_dataloader = DataLoader(val_data, batch_size=batchsize)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info("-------------------------{}th train------------------------\n"
                    "the number of pictures of train set: {}\n"
                    "the number of pictures of validation set: {}\n".format(i, len(train_img), len(val_img)))

        train_step = 0
        best_train_iou = 0
        best_test_iou = 0
        for n in range(epoch):
            model.train()
            for data in train_dataloader:
                imgs, masks = data
                predicts = model(imgs.to(device))

                # 反向传播
                optimizer.zero_grad()
                masks = masks.squeeze(dim=1).type(torch.long).to(device)
                loss = loss_fn(predicts, masks)
                loss.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()

                if train_step % 10 == 0:
                    logger.info("after {} steps training, the loss is {:.4f}".format(train_step, loss))
                train_step += 1

            # 验证集
            model.eval()
            metric.reset()
            with torch.no_grad():
                for data in val_dataloader:
                    imgs, masks = data
                    predicts = model(imgs.to(device))
                    # 指标计算
                    predicts = torch.argmax(predicts, dim=1)
                    metric.addBatch(predicts.cpu(), masks.cpu())

                # class_acc = metric.classPixelAccuracy()
                acc = metric.meanPixelAccuracy()
                # class_iou = metric.classIntersectionOverUnion()
                iou = metric.meanIntersectionOverUnion()
                # class_dice = metric.classDice_score()
                dice = metric.Dice_score()
                # print(num_classes)
                # print(class_acc)
                # logger.info("val:       background      cancer      benign")
                # logger.info("val: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))
                # logger.info("val:ACC    {:.4f}          {:.4f}      {:.4f}".format(*class_acc))
                # logger.info("val:IOU    {:.4f}          {:.4f}      {:.4f}".format(*class_iou))
                # logger.info("val:DICE   {:.4f}          {:.4f}      {:.4f}".format(*class_dice))
                logger.info("val: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))

            if iou > best_train_iou:
                best_train_iou = iou
                train_best = [acc, iou, dice]
                # 保存best_iou的模型参数
                save_path = "weights/{}_{}fold_{}epoch_{:.4f}ACC_{:.4f}IOU_{:.4f}DICE.pth".format(model_name, i, n, acc,
                                                                                                  iou, dice)
                torch.save(model, save_path)
                logger.info("save_path: {}".format(save_path))
                # 内部测试集
                if inter_log != None:
                    temp_test = seg_test(read_log=inter_log, model_pth=save_path, write_log=record_log,
                                         task=task, device=device)
                    if temp_test[1] > best_test_iou:
                        best_test_iou = temp_test[1]
                        test_best = temp_test

        save_checkpoint_model(i,model, optimizer, loss, None, checkpoint_path)


        # 保存每折最佳
        train_value.append(train_best)
        test_value.append(test_best)
        logger.info(f"{'-' * 20} {model_name} {i}fold best  performance {'-' * 20}")
        logger.info("VAL: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*train_best))
        logger.info("TEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*test_best))

    #  最佳值的均值
    val_mean = np.mean(np.array(train_value), axis=0)
    test_mean = np.mean(np.array(test_value), axis=0)
    logger.info(f"{'-' * 20} {task} {model_name}  avg performance {'-' * 20}")
    logger.info("VAL: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*val_mean))
    logger.info("TEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*test_mean))
    logger.info("-" * 50)



