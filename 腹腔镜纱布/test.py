import pandas as pd
import torch.optim
from torch.utils.data import DataLoader

from Model.Detection import SSD, faster_rcnn
from Model.SematicSegment import Unet
from data_process.dataset import *
from metric_function.seg_metric import SegmentationMetric
from metric_function.detection_metric import DetectionMetrics, get_TrueBox_and_predBoex
from log import logger_config


def seg_test(read_log, model_pth, write_log, device):
    """

    :param log_file:
    :param model_pth:
    :return:
    """
    # 获取数据集
    image_root = "data/images"
    mask_root = "data/mask"
    batchsize = 1
    test_img, test_mask = get_data(log_file=read_log, mask_root=mask_root, task="seg")
    train_data = seg_dataset(data_image=test_img, data_mask=test_mask,
                                 image_root=image_root, mask_root=mask_root)    
    test_dataloader = DataLoader(train_data, batch_size=batchsize)
    logger = logger_config(log_path=write_log, logging_name="{} {}".format(model_pth, read_log))
    logger.info(f"the number of pictures of test set: {len(test_img)}")
    # 加载模型测试
    model = torch.load(model_pth,map_location=device)
    metric = SegmentationMetric(numClass=2)
    i = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, masks = data
            predicts = model(imgs.to(device))
            if "COD" in model_pth:
                predicts = predicts[3].sigmoid().data.squeeze()
                predicts = (predicts - predicts.min()) / (predicts.max() - predicts.min() + 1e-8)
            else:
                predicts = torch.argmax(predicts, dim=1)
            if predicts.shape != masks.shape:
                predicts = predicts.unsqueeze(dim=0)
            predicts,masks = predicts.cpu(), masks.cpu()
            metric.reset()
            metric.addBatch(predicts, masks)
            acc = metric.meanPixelAccuracy()
            iou = metric.meanIntersectionOverUnion()
            dice = metric.Dice_score()
            logger.info("img_id: {} test set: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(test_img[i],acc, iou, dice))
            i += 1
    # acc = metric.meanPixelAccuracy()
    # iou = metric.meanIntersectionOverUnion()
    # dice = metric.Dice_score()

    # logger.info("{} test {}".format('-'*20, '-'*20))
    # logger.info("test set: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))
    return [acc, iou, dice]


def detection_test(read_log, model_pth, write_log, device):
    # 获取数据集
    image_root = "data/images"
    mask_root = "data/mask"
    json_root = "data/json"
    batchsize = 8
    test_img, test_mask = get_data(log_file=read_log, mask_root=mask_root, task="detection")
    test_data = detection_dataset(data_image=test_img, data_json=test_mask,
                                  image_root=image_root, json_root=json_root)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, collate_fn=my_collate)
    logger = logger_config(log_path=write_log, logging_name="{} {}".format(model_pth, read_log))
    logger.info(f"the number of pictures of test set: {len(test_img)}")
                    
    model = torch.load(model_pth).to(device)
    metric = DetectionMetrics()
    model.eval()
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            predicts = model(imgs.to(device))
            pred_boxes, true_boxes = get_TrueBox_and_predBoex(targets, predicts)
            metric.addBatch(pred_boxes, true_boxes)
    mAP50 = metric.calculate_map(iou_threshold=0.5)
    precision, recall = metric.calculate_precision_recall(iou_threshold=0.4)

    logger = logger_config(log_path=write_log, logging_name="{} {}".format(model_pth, read_log))
    logger.info("{} test {}".format('-' * 20, '-' * 20))
    logger.info("test: mAP50_{:.4f} precision_{:.4f} recall_{:.4f}".format(mAP50, precision, recall))
    return [mAP50, precision, recall]

