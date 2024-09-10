# -*- coding=utf-8 -*-
# @TIME 2023/11/22 22:16
# @Author: lyl
# @File: train.py
# @Software:PyCharm

import torch.optim
from Model import fcn_resnet, Unet, Network_Res2Net_GRA_NCD, FCT_exp
from Model.Network_Res2Net_GRA_NCD import structure_loss
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


def seg_train(model_name, fold, device, task,model_type=None, continue_train=False):
    """

    :param model_name:
    :param fold:
    :param device:
    :param task: liver_seg or one_step_seg or two_step_seg
    :param model_type:
    :param continue_train:
    :return:
    """

    # 模型选择
    if model_name == "fcn_resnet":
        model = fcn_resnet.fcn(num_classes=2, model_type=model_type)
        model_name = "{}{}".format(model_name, model_type)
    elif model_name == "UNet":
        model = Unet.UNet(in_channels=3, num_classes=2)
    elif model_name == "COD":
        model = Network_Res2Net_GRA_NCD.Network(imagenet_pretrained=True)
        # model = Network_Res2Net_GRA_NCD.Network(imagenet_pretrained=False)
        # model.load_state_dict(torch.load("weights/Net_epoch_best.pth"))
    elif model_name == "FCT":
        model = FCT_exp.FCT(in_channels=3, num_classes=2)
    else:
        print(f"There is no {model_name} model...")
    model = model.to(device)

    # 设置参数
    record_log = "log/seg_log.txt"
    image_root = "data/images"
    mask_root = "data/mask"

    logger = logger_config(log_path=record_log, logging_name=f"{task} {model_name} train")
    loss_fn = nn.CrossEntropyLoss().to(device)
    metric = SegmentationMetric(numClass=2)
    train_value = []
    test_value = []
    learning_rate = 1e-5
    epoch = 120
    batchsize = 12
    checkpoint_path = "checkpoint.pth"

    for i in range(fold):
        train_str = "log/{} fold train.txt".format(i)
        val_str = "log/{} fold val.txt".format(i)
        train_img, train_mask = get_data(log_file=train_str, mask_root=mask_root, img_root=image_root)
        val_img, val_mask = get_data(log_file=val_str, mask_root=mask_root, img_root=image_root)
        train_data = img_dataset(data_image=train_img, data_mask=train_mask)
        val_data = img_dataset(data_image=val_img, data_mask=val_mask)
        train_dataloader = DataLoader(train_data, batch_size=batchsize)
        val_dataloader = DataLoader(val_data, batch_size=batchsize)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info("------------------------- {}th train ------------------------\n"
                    "the number of pictures of train set: {}\n"
                    "the number of pictures of validation set: {}\n".format(i, len(train_img), len(val_img)))

        train_step = 0
        best_train_iou = 0
        best_test_iou = 0
        for n in range(epoch):
            if continue_train and os.path.exists(checkpoint_path):
                model, optimizer, i, n, loss = load_checkpoint_model(checkpoint_path, model, optimizer)
                continue_train = False
            model.train()
            for data in train_dataloader:
                imgs, masks = data
                predicts = model(imgs.to(device))

                # 反向传播
                optimizer.zero_grad()
                if model_name == "COD":
                    masks = masks.unsqueeze(dim=1).to(device)
                    loss_init = structure_loss(predicts[0], masks) + structure_loss(predicts[1],masks) + structure_loss(predicts[2],                                                                                       masks)
                    loss_final = structure_loss(predicts[3], masks)
                    loss = loss_final + loss_init
                elif model_name == "FCT":
                    masks = masks.type(torch.long).to(device)
                    loss = loss_fn(predicts[2], masks)
                else:
                    masks = masks.squeeze(dim=1).type(torch.long).to(device)
                    loss = loss_fn(predicts, masks)
                loss.backward()
                clip_gradient(optimizer, 0.45)
                optimizer.step()

                if train_step % 100 == 0:
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
                    if model_name == "COD":
                        predicts = predicts[3].sigmoid().data.squeeze()
                        predicts = (predicts - predicts.min()) / (predicts.max() - predicts.min() + 1e-8)
                    else:
                        predicts = torch.argmax(predicts, dim=1)
                    if predicts.shape != masks.shape:
                        predicts = predicts.unsqueeze(dim=0)
                    metric.addBatch(predicts.cpu(), masks.cpu())

                acc = metric.meanPixelAccuracy()
                iou = metric.meanIntersectionOverUnion()
                dice = metric.Dice_score()
                logger.info("val: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))

            if iou > best_train_iou:
                best_train_iou = iou
                train_best = [acc, iou, dice]
                # 保存best_iou的模型参数
                save_path = "weights/{}_{}fold_{}epoch_{:.4f}ACC_{:.4f}IOU_{:.4f}DICE.pth".format(model_name, i, n, acc,
                                                                                                  iou, dice)
                torch.save(model, save_path)
                logger.info("save_path: {}".format(save_path))
                # # 内部测试集
                # if inter_log != None:
                #     temp_test = seg_test(read_log=inter_log, model_pth=save_path, write_log=record_log, task=task,
                #                          device=device)
                #     if temp_test[1] > best_test_iou:
                #         best_test_iou = temp_test[1]
                #         test_best = temp_test
                # save_checkpoint_model(i, n, model, optimizer, loss, None, checkpoint_path)

        # 保存每折最佳
        train_value.append(train_best)
        # test_value.append(test_best)
        logger.info(f"{'-' * 20} {model_name} best  performance {'-' * 20}")
        logger.info("VAL: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*train_best))
        # logger.info("TEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*test_best))

    #  最佳值的均值
    val_mean = np.mean(np.array(train_value), axis=0)
    # test_mean = np.mean(np.array(test_value), axis=0)
    logger.info(f"{'-' * 20} {model_name}  avg performance {'-' * 20}")
    logger.info("VAL: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*val_mean))
    # logger.info("TEST: ACC:{:.4f} IOU:{:.4f} DICE:{:.4f}".format(*test_mean))
    logger.info("-" * 50)



