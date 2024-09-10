 
from train import *
import torch
from test import *


"""----------------------训练时运行该部分--------------------"""
# #二分类 # #
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
fold = 5
task = "liverrupture_surgery"
# classification_train(task=task, fold=fold, model_name="Densenet", device=device)
# classification_train(task=task, fold=fold, model_name="CNN", device=device)
# classification_train(task=task, fold=fold, model_name="Resnet", model_type=101, device=device)
# classification_train(task=task, fold=fold, model_name="Resnet", model_type=18, device=device)

"""-----------------------------------------------------"""
# save_path = "weights/liverrupture_surgery_Resnet_18_1fold_2epoch_1.0000AUC_1.0000ACC.pth"
# save_path = "weights/liverrupture_surgery_Densenet_4fold_14epoch_0.9852AUC_0.9385ACC.pth"
save_path = "weights/liverrupture_surgery_Densenet_4fold_27epoch_0.9211AUC_0.8923ACC.pth"
log_intest_data = f"log/{task}_test.txt"
img_root = "/home/user/user/2/injury/data/images"
test_temp = classification_test(log_file=log_intest_data, model_pth=save_path,
                            data_root=img_root, writeLog="log.txt", device=device)