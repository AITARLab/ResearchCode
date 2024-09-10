import torch.optim
from torch.utils.data import DataLoader
from data_process.dataset import *
from metric_function.seg_metric import SegmentationMetric
from log.log import logger_config


def seg_test(read_log, model_pth, write_log, device):
    """

    :param log_file:
    :param model_pth:
    :return:
    """
    # 获取数据集
    image_root = "data/image"
    mask_root = "data/mask"
    batchsize = 4
    test_img, test_mask = get_data(log_file=read_log, mask_root=mask_root, img_root=image_root)
    test_data = img_dataset(data_image=test_img, data_mask=test_mask)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)
    logger = logger_config(log_path=write_log, logging_name="{} {}".format(model_pth, read_log))

    # 加载模型测试
    model = torch.load(model_pth).to(device)
    metric = SegmentationMetric(numClass=7)
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, masks = data
            predicts = model(imgs.to(device))
            predicts = torch.argmax(predicts, dim=1)
            predicts, masks = predicts.cpu(), masks.cpu()
            metric.addBatch(predicts, masks)

    logger.info("{} test {}".format('-' * 20, '-' * 20))
    classAcc = metric.classPixelAccuracy()  # 每个类别的acc
    acc = metric.meanPixelAccuracy()  # 总共的acc
    classiou = metric.classIntersectionOverUnion()
    iou = metric.meanIntersectionOverUnion()
    classdice = metric.classDice_score()
    dice = metric.Dice_score()
    logger.info('background":0, "股内侧肌":1, "缝匠肌":2, "半膜肌":3,"半腱肌":4,"股二头肌":5,"股薄肌":6,"骨薄肌":6')
    logger.info("val: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))
    logger.info(f"val: classACC {classAcc} classIOU {classiou} classDICE {classdice}")

    return [acc, iou, dice]
