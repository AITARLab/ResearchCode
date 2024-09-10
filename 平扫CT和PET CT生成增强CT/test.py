import torch.optim
from torch.utils.data import DataLoader
from data_process.dataset import *
from metric_function.seg_metric import SegmentationMetric
from log.log import logger_config


def seg_test(read_log, model_pth, write_log, task, device):
    """

    :param log_file:
    :param model_pth:
    :return:
    """
    # 获取数据集
    image_root = "data/image"
    mask_root = f"data/{task}_mask"
    batchsize = 8
    test_img, test_mask = get_data(log_file=read_log, mask_root=mask_root, img_root=image_root)
    test_data = img_dataset(data_image=test_img, data_mask=test_mask, task=task)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)
    logger = logger_config(log_path=write_log, logging_name="{} {}".format(model_pth, read_log))

    # 加载模型测试
    model = torch.load(model_pth).to(device)
    num_classes = 2
    metric = SegmentationMetric(numClass=num_classes)
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, masks = data
            predicts = model(imgs.to(device))
            predicts = torch.argmax(predicts, dim=1)
            predicts, masks = predicts.cpu(), masks.cpu()
            metric.addBatch(predicts, masks)

    logger.info("{} test {}".format('-' * 20, '-' * 20))
    # class_acc = metric.classPixelAccuracy()
    acc = metric.meanPixelAccuracy()
    # class_iou = metric.classIntersectionOverUnion()
    iou = metric.meanIntersectionOverUnion()
    # class_dice = metric.classDice_score()
    dice = metric.Dice_score()
    # logger.info("test:       background      cancer      benign")
    # logger.info("test:ACC    {:.4f}          {:.4f}      {:.4f}".format(*class_acc))
    # logger.info("test:IOU    {:.4f}          {:.4f}      {:.4f}".format(*class_iou))
    # logger.info("test:DICE   {:.4f}          {:.4f}      {:.4f}".format(*class_dice))
    logger.info("test:ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))
    logger.info("{} test {}".format('-' * 20, '-' * 20))
    return [acc, iou, dice]
