import os.path
from draw_analysis import draw_mask, draw_rectangle
import torch
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib_inline
import numpy as np
from torchvision import transforms
import torchvision.models as models
import numpy as np
import json


def GT_mask_draw(img_path, mask_path):
    # 输入图像处理
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    mask = np.array(Image.open(mask_path))/255
    mask_array = mask[:, :, np.newaxis]
    img_with_mask = draw_mask(img_array, mask_array, color='green', alpha=0.8)
    save_path = "GT.png"
    plt.imsave(save_path, img_with_mask)


def GT_detection_draw(img_root,json_root):
    for i in range(0, 36):
        img_id = f"V11_{i}.png"
        img_path = os.path.join(img_root, img_id)
        json_path = os.path.join(json_root, img_id.split(".")[0]+".json")
        if os.path.exists(json_path) == False:
            continue
        img = Image.open(img_path).convert('RGB')

        with open(json_path, encoding='UTF-8') as f:
            content = json.load(f)
        shapes = content['shapes']
        for shape in shapes:
            category = shape['label']
            if "rectangle" in category:
                points = shape['points']
                x_left = min(points[0][0], points[1][0])
                x_right = max(points[0][0], points[1][0])
                y_left = min(points[0][1], points[1][1])
                y_right = max(points[0][1], points[1][1])
                temp = [x_left, y_left, x_right, y_right]
                img = draw_rectangle(img, tuple(temp), "green", 8)
        save_path = "results/{}_detection_GT.png".format(img_id.split(".")[0])
        img.save(save_path)


def seg_inference(model_path, img_path,save_path):
    """

    :param model_path:
    :param img_path:
    :return:
    """
    # 加载模型
    model = torch.load(model_path, map_location='cpu')
        # 输入图像处理
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    w, h = img.size[0], img.size[1]
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # 预测结果
    pred = model(trans(img_array).unsqueeze(dim=0))
    if "COD" in model_path:
        pred = pred[3].sigmoid().data.squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = pred.unsqueeze(dim=0)
    else:
        pred = torch.argmax(pred, dim=1)

    # print(pred.shape)
    pred = transforms.Resize([h, w])(pred)
    # 制图
    pred = pred.squeeze(dim=0).numpy()
    mask_array = pred[:, :, np.newaxis]
    img_with_mask = draw_mask(image=img_array, mask=mask_array, color='green', alpha=0.8)
    # plt.imshow(img_with_mask)
    # plt.show()
    # save_path = "pred.png"
    plt.imsave(save_path, img_with_mask)


def detection_inference(model_path=None, img_root=None):
    # 加载模型
    model = torch.load(model_path, map_location='cpu')
    # 输入图像处理
    model.eval()
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                transforms.Normalize([0.48235, 0.45882, 0.40784],
                                                     (0.229, 0.224, 0.225))])
    for i in range(10, 36):
        img_id = f"V11_{i}.png"
        img_path = os.path.join(img_root, img_id)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        x, y = img_array.shape[1], img_array.shape[0]
        # print(img_array.shape)
        # 运行模型推理
        predictions = model(trans(img_array).unsqueeze(0))
        # plt.imshow(img_array)
        # plt.show()
        # 解析预测结果
        for det in predictions[0]['boxes']:
            box = det.detach().numpy() / 512
            # print([box[0]*x, box[1]*y, box[2]*x, box[3]*y])
            img = draw_rectangle(img, tuple([box[0]*x, box[1]*y, box[2]*x, box[3]*y]), "yellow", 8)
        save_path = "results/{}_detection_pred.png".format(img_id.split(".")[0])
        img.save(save_path)
        # plt.imshow(img)
        # plt.show()









# model_pth = './weights/fcn_resnet50_3fold_5epoch_0.8944ACC_0.8575IOU_0.9183DICE.pth'
# img_root = r"data\image"
# mask_root = r"data\mask"
# json_root = r"data\json"
# model_pth = "weights/COD_3fold_5epoch_0.9738ACC_0.9693IOU_0.9842DICE.pth"
# seg_inference(model_path=model_pth, img_root=img_root)
# GT_draw(img_root, mask_root)
# GT_detection_draw(img_root, json_root)
# detection_inference(model_path=model_pth, img_root=img_root)