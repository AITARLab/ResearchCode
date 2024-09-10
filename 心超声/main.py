from data_process.preproecess import *
from data_process.dataset import *
from train import seg_train

root = r"C:\桌面\AI\focus\心超声\原始数据"
img_root = "data/images"
json_root = "data/json"
mask_root = "data/mask"
# clean_data(root, img_root, json_root)
# getmask(img_root, json_root, mask_root)
# split_data(data_root= img_root, fold=5)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
fold = 5
# seg_train(model_name="UNet", fold=fold, device=device, task="seg1")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=50, task="seg1")
# seg_train(model_name="fcn_resnet", fold=fold, device=device, model_type=101, task="seg1")
# seg_train(model_name="COD", fold=fold, device=device, task="seg1")
seg_train(model_name="FCT", fold=fold, device=device, task="seg1")