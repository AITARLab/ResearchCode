import torch.optim
from torch.utils.data import DataLoader
from data_process.dataset import *
from metric_function.seg_metric import SegmentationMetric
from log import logger_config


def seg_test(read_log, model_pth, write_log, device, task):
    """

    :param read_log:
    :param model_pth:
    :param write_log:
    :param device:
    :param task:
    :return:
    """
    # 获取数据集
    if task == "spleen_seg":
        image_root = "data/images"
        mask_root = "data/spleen_mask"
    elif task == "one_step_seg":
        image_root = "data/images"
        mask_root = "data/spleenrupture_mask"
    elif task == "two_step_seg":
        image_root = "data/predict_spleen"
        mask_root = "data/spleenrupture_mask"

    batchsize = 8
    test_img, test_mask = get_data(log_file=read_log, mask_root=mask_root, img_root=image_root)
    test_data = img_dataset(data_image=test_img, data_mask=test_mask)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)
    logger = logger_config(log_path=write_log, logging_name="{} {}".format(model_pth, read_log))

    # 加载模型测试
    model = torch.load(model_pth).to(device)
    metric = SegmentationMetric(numClass=2)
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
            predicts, masks = predicts.cpu(), masks.cpu()
            metric.addBatch(predicts, masks)
    acc = metric.meanPixelAccuracy()
    iou = metric.meanIntersectionOverUnion()
    dice = metric.Dice_score()

    logger.info("{} test {}".format('-' * 20, '-' * 20))
    logger.info("test set: ACC {:.4f} IOU {:.4f} DICE {:.4f}".format(acc, iou, dice))
    return [acc, iou, dice]
