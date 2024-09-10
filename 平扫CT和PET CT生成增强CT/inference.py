import os.path
from log.draw_analysis import draw_mask, draw_rectangle
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
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    mask = np.load(mask_path)
    mask_array = mask[:, :, np.newaxis]
    img_with_mask = draw_mask(img_array, mask_array, color='green', alpha=0.8)
    save_path = "results/GT.png"
    plt.imsave(save_path, img_with_mask)


def seg_inference(model_path, img_path):
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
    pred = torch.argmax(pred, dim=1)
    pred = transforms.Resize([h, w])(pred)
    # 制图
    pred = pred.squeeze(dim=0).numpy()
    mask_array = pred[:, :, np.newaxis]
    img_with_mask = draw_mask(image=img_array, mask=mask_array, color='yellow', alpha=0.8)
    save_path = "results/pred.png"
    plt.imsave(save_path, img_with_mask)


