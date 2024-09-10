from data_process.preprocess import *
from yolo import *
import torch
from train import *
from test import *
from metric_function.yolo_metric import *
from standardization.refine import get_rectangle_id
from standardization.new_data import new_test_generation, test_log_refine, new_images, new_mask_and_label

# rename("F:\腹腔镜手术纱布识别\术中识别纱条（图片）\dataset", 'data')
# get_label("data/json")
# split_data(data_root = "data/images",total_number=30, exter_test=[[30],[31]], fold=5)
# root = "log"
# for file in os.listdir(root):
#     path = root + "/" + file
#     get_yolo_data(path)

fold = 5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
"""---------------- detection -----------"""
# yolo_train(fold=fold)
pth = "runs/detect/train222/weights/best.pt"
yolo_eval(pth=pth)
# detection_train(model_name="SSD", device=device, fold=fold)
# detection_train(model_name="faster_rcnn", device=device, fold=fold, model_type=50)
# detection_train(model_name="faster_rcnn", device=device, fold=fold, model_type="mobilenet")
# pth = "weights/faster_rcnn_mobilenet_3fold_0epoch_1.0000mAP50_1.0000precision_1.0000recall.pth"
# detection_test(read_log="log/inter_set.txt", model_pth=pth, write_log="log/detection+lob.txt", device=device)
"""---------------- segmentation -----------"""
# seg_train(model_name="COD", fold=fold, device=device)
# seg_train(model_name="UNet", fold=fold, device=device)
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=50)
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=101)
# pth="weights/fcn_resnet101_4fold_1epoch_0.9722ACC_0.9558IOU_0.9770DICE.pth"
# print(device)
# seg_test(read_log="log/exter_set_1.txt", model_pth=pth, write_log="log/seg_log.txt", device=device)
"""--------------standardization--------"""
# root = "data/json"
# for file in os.listdir(root):
#     json_path = root + "/" +file
#       get_rectangle_id(json_path)

# for file in ["log/inter_set.txt","log/exter_set_0.txt", "log/exter_set_1.txt"]:
#     new_test_generation(file)
#     test_log_refine(file)
# list1 = os.listdir("standardization/data/json")
# list2 = os.listdir("data/json")
# print(len(list1), len(list2))
# get_json(yolo_json="runs/detect/val2/predictions.json",iou=0.4)